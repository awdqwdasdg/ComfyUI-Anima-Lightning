"""ComfyUI custom node: Anima TDM-Unify (k branch).

Injects the 4 TDM-Unify tensors (step-count time conditioning) of
Anima-Lightning (https://huggingface.co/aina-tech/Anima-Lightning) into
converted Anima / Cosmos-Predict2 style models in ComfyUI, while keeping the
base checkpoint and any LoRAs extracted from it fully compatible.

Anima-Lightning adds a small "k branch" to the timestep embedding that
conditions the model on the number of sampling steps K:

    k_cond            = shift / (K + shift - 1)
    temb              = temb + k_scale * k_MLP(k_freq)
    embedded_timestep = RMSNorm(t_freq) + k_scale * k_norm(k_freq)

The ComfyUI conversion of the checkpoint drops these tensors, so they are
re-injected at runtime from `anima_tdm_k_branch.safetensors` (~34 MB). That
file is obtained automatically: downloaded on first use (sha256-pinned,
cached next to this node), unless a copy is placed next to this node or
anywhere under ComfyUI/models/diffusion_models/. ANIMA_TDM_K_PACK_URL can
point the download at a mirror.

Determinism: shims are always rebuilt from pristine modules tracked per
diffusion model, never from whatever happens to be installed on the shared
module tree, so re-running this node is idempotent - patching cannot stack
across workflow hops, cache evictions or re-executions, and bypassing the
node restores the exact base model. Only `diffusion_model.t_embedder.1` and
`diffusion_model.t_embedding_norm` are object-patched, via same-class
shallow copies, so ComfyUI's weight-patching / manual-cast / lowvram /
quantized-weight machinery and all LoRA key paths keep working unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import threading
import types
import urllib.request
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import folder_paths  # running inside ComfyUI
except Exception:  # pragma: no cover - tests / standalone inspection
    folder_paths = None

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PACK_FILENAME = "anima_tdm_k_branch.safetensors"
SIDECAR_PATH = os.path.join(NODE_DIR, PACK_FILENAME)

# The extra tensors file is downloaded automatically on first use and cached
# next to this node. ANIMA_TDM_K_PACK_URL can redirect the download to a
# mirror; the sha256 pin applies to the default URL only.
_DEFAULT_PACK_URL = "https://huggingface.co/awdqwdasdg/Anima-Lightning-Comfyui/resolve/main/anima_tdm_k_branch.safetensors"
_DEFAULT_PACK_SHA256 = "6fae67d2533503ec5dd28230e051203ad955c8a9087d5079d20bcccecc3294c4"
_PACK_URL = os.environ.get("ANIMA_TDM_K_PACK_URL") or _DEFAULT_PACK_URL
_PACK_SHA256 = _DEFAULT_PACK_SHA256 if _PACK_URL == _DEFAULT_PACK_URL else ""

# tdm_unify_shift from the Anima-Lightning transformer config. This also
# matches the `shift: 3.0` sampling setting ComfyUI uses for the Anima model
# family, so k_cond == the shifted final nonterminal sigma of the schedule.
TDM_SHIFT = 3.0
# RMSNorm epsilon of the k branch's own norm (diffusers RMSNorm default used
# by CosmosEmbedding; the pack carries no eps metadata).
K_NORM_EPS = 1e-6
DEFAULT_STEPS = 4  # Anima-Lightning official recipe (4 steps, CFG 1.0)

PACK_KEYS = (
    "time_embed.k_embed.norm.weight",
    "time_embed.k_embed.t_embedder.linear_1.weight",
    "time_embed.k_embed.t_embedder.linear_2.weight",
    "time_embed.k_scale",
)
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


# --------------------------------------------------------------------------
# extra tensors loading
# --------------------------------------------------------------------------

def _read_pack(path):
    """Minimal, dependency-free safetensors reader for the 4-tensor pack."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        tensors = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype = _DTYPES.get(meta.get("dtype"))
            if dtype is None:
                raise ValueError("unsupported dtype %r for %s" % (meta.get("dtype"), key))
            start, end = meta["data_offsets"]
            f.seek(8 + n + start)
            raw = f.read(end - start)
            t = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
            shape = meta.get("shape") or []
            tensors[key] = (t.reshape(shape) if shape else t.reshape(())).contiguous()
    return tensors


def _cache_key(path):
    try:
        st = os.stat(path)
        return (path, st.st_mtime, st.st_size)
    except OSError:
        return None


_K_CACHE = {}      # (path, mtime, size) -> {key: tensor}
_PACK_SCAN = {"found": None}


def _model_pack_dirs():
    dirs = []
    if folder_paths is not None:
        for name in ("diffusion_models", "unet"):
            try:
                dirs.extend(folder_paths.get_folder_paths(name))
            except Exception:
                pass
    return [d for d in dict.fromkeys(dirs) if isinstance(d, str) and os.path.isdir(d)]


def _find_pack_in_model_dirs():
    if _PACK_SCAN["found"] and os.path.isfile(_PACK_SCAN["found"]):
        return _PACK_SCAN["found"]
    for root_dir in _model_pack_dirs():
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = dirnames[:16]  # bounded fan-out
            if dirpath[len(root_dir):].count(os.sep) >= 3:
                dirnames[:] = []         # bounded depth
            if PACK_FILENAME in filenames:
                found = os.path.join(dirpath, PACK_FILENAME)
                _PACK_SCAN["found"] = found
                return found
    return None


def _download_url_file(url, dest, expected_sha256):
    """Atomic download; verifies the pinned hash when one is given."""
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "comfyui-anima-tdm-unify"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if expected_sha256:
            h = hashlib.sha256()
            with open(tmp, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            if h.hexdigest() != expected_sha256:
                raise ValueError(
                    "SHA-256 mismatch: got %s, expected %s"
                    % (h.hexdigest(), expected_sha256)
                )
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _missing_pack_error(failures):
    """Error for "the extra tensors file could not be obtained"."""
    lines = [
        "Anima TDM-Unify: anima_tdm_k_branch.safetensors (the 4-tensor "
        "TDM-Unify extra tensors file) could not be obtained.",
        "It is normally downloaded automatically on first use and cached "
        "next to this node. It can also be provided manually:",
        "  1. copy it next to this node:  %s" % NODE_DIR,
        "  2. put it anywhere under:      ComfyUI/models/diffusion_models/",
        "Direct URL: %s" % _PACK_URL,
    ]
    if _PACK_SHA256:
        lines.append("SHA-256: %s" % _PACK_SHA256)
    if failures:
        lines += ["", "Failure details:"] + failures
    return RuntimeError("\n".join(lines))


def _resolve_k_tensors():
    """Return ({key: cpu tensor}, source label) for the 4 k-branch tensors."""
    failures = []
    candidates = []
    if os.path.isfile(SIDECAR_PATH):
        candidates.append(("next to node", SIDECAR_PATH, False))
    found = _find_pack_in_model_dirs()
    if found:
        candidates.append(("models dir", found, False))
    candidates.append(("download", None, True))
    for label, path, is_url in candidates:
        try:
            if is_url:
                if not os.path.isfile(SIDECAR_PATH):
                    _download_url_file(_PACK_URL, SIDECAR_PATH, _PACK_SHA256)
                path, label = SIDECAR_PATH, "downloaded (cached)"
            key = _cache_key(path)
            if key is None:
                continue
            if key not in _K_CACHE:
                _K_CACHE[key] = _read_pack(path)
            return _K_CACHE[key], label
        except Exception as exc:  # try the next source
            failures.append("  - %s failed: %r" % (label, exc))
    raise _missing_pack_error(failures)


def _pack_fingerprint():
    """Stable fingerprint of the pack actually in use (for IS_CHANGED)."""
    try:
        if os.path.isfile(SIDECAR_PATH):
            path = SIDECAR_PATH
        else:
            path = _find_pack_in_model_dirs() or ""
        if path and os.path.isfile(path):
            st = os.stat(path)
            return (path, st.st_mtime, st.st_size)
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------
# k branch state
# --------------------------------------------------------------------------

class _KBranch:
    """Per-node-execution k branch: raw weights + conditioning source.

    Holds only immutable inputs plus lazily-cast weight copies; it is shared
    by the two module shims of one patched model and is never mutated
    in-place from the forward path, so re-running the node produces a
    functionally identical branch.
    """

    def __init__(self, tensors, timesteps_module, steps, auto_k, shift=TDM_SHIFT):
        self._raw = {
            "norm": tensors["time_embed.k_embed.norm.weight"],
            "l1": tensors["time_embed.k_embed.t_embedder.linear_1.weight"],
            "l2": tensors["time_embed.k_embed.t_embedder.linear_2.weight"],
            "scale": tensors["time_embed.k_scale"].reshape(()),
        }
        self._timesteps = timesteps_module  # the model's own Timesteps (frequency parity)
        self._steps = int(steps)
        self.auto_k = bool(auto_k)
        self._shift = float(shift)
        self._cast = {}
        # Live step count, set per-thread by the model_function_wrapper right
        # before each model call (thread-local => safe with parallel model
        # call threads).
        self._tl = threading.local()

    # -- conditioning ------------------------------------------------------

    def _live_steps(self):
        if self.auto_k:
            k = getattr(self._tl, "live_steps", None)
            if isinstance(k, int) and k >= 1:
                return k
        return self._steps if self._steps >= 1 else 1

    def k_cond(self):
        """shifted_reciprocal_condition(K) = shift / (K + shift - 1)."""
        k = self._live_steps()
        return self._shift / (k + self._shift - 1.0)

    def set_live_sigmas(self, sigmas):
        """Record the sampler's actual step count for this thread."""
        try:
            k = int(sigmas.numel()) - 1  # steps = transitions between sigmas
        except Exception:
            return
        if k >= 1:
            self._tl.live_steps = k

    def clear_live_sigmas(self):
        self._tl.live_steps = None

    # -- weights -------------------------------------------------------------

    def _weights(self, device, dtype):
        key = (device, dtype)
        w = self._cast.get(key)
        if w is None:
            w = tuple(self._raw[name].to(device=device, dtype=dtype)
                      for name in ("norm", "l1", "l2", "scale"))
            self._cast[key] = w
        return w

    # -- k terms (each computed exactly once per call site) ------------------

    def _k_freq(self, ref):
        """Sinusoidal features of the constant k_cond, via the model's own
        Timesteps module so the frequency layout matches exactly."""
        b, t = ref.shape[0], ref.shape[1]
        cond = self.k_cond()
        inp = torch.full((b, t), cond, dtype=torch.float32, device=ref.device)
        return self._timesteps(inp)  # (B, T, D) float32

    def k_embedded(self, ref):
        """k_scale * k_norm-weighted RMSNorm(k_freq): added to the output of
        t_embedding_norm (the embedded_timestep path)."""
        w_norm, _, _, _ = self._weights(ref.device, torch.float32)
        freq = self._k_freq(ref)
        var = freq.pow(2).mean(-1, keepdim=True)
        out = freq * torch.rsqrt(var + K_NORM_EPS) * w_norm
        _, _, _, w_scale = self._weights(ref.device, ref.dtype)
        return w_scale * out.to(ref.dtype)

    def k_temb(self, ref):
        """k_scale * k_MLP(k_freq): added to the AdaLN-LoRA timestep
        projection (the temb path)."""
        _, l1, l2, w_scale = self._weights(ref.device, ref.dtype)
        freq = self._k_freq(ref).to(ref.dtype)
        out = F.linear(F.silu(F.linear(freq, l1)), l2)
        return w_scale * out


# --------------------------------------------------------------------------
# pristine module tracking (the determinism fix)
# --------------------------------------------------------------------------

_STATE_ATTR = "_anima_tdm_k_state"
_PRISTINE = weakref.WeakKeyDictionary()  # diffusion_model -> {"t1": ..., "norm": ...}


def _unwrap_chain(module):
    """Walk old-node `_orig` chains down to the pristine module, and rebuild
    a pristine-behaving module out of one of *our* tagged copies."""
    cur = module
    for _ in range(32):  # sanity bound on chain length
        nxt = getattr(cur, "_orig", None)  # original node's shims keep _orig
        if isinstance(nxt, nn.Module):
            cur = nxt
            continue
        break
    if getattr(cur, _STATE_ATTR, None) is not None:
        # one of our shallow-copy shims: its children/parameters are shared
        # with the pristine module, so a copy without the tag and without the
        # instance-level forward behaves exactly like the pristine module.
        fresh = copy.copy(cur)
        fresh.__dict__.pop(_STATE_ATTR, None)
        fresh.__dict__.pop("forward", None)  # fall back to the class forward
        cur = fresh
    return cur


def _pristine_record(dm):
    """Return (and memoize per diffusion model) the pristine t_embedder[1]
    and t_embedding_norm modules, regardless of what is currently installed
    on the shared module tree."""
    rec = _PRISTINE.get(dm)
    if rec is not None:
        return rec

    t_embedder = getattr(dm, "t_embedder", None)
    t1 = _unwrap_chain(t_embedder[1])
    norm = _unwrap_chain(getattr(dm, "t_embedding_norm", None))

    # Migration: if the original node's Sequential shim is currently
    # installed as `t_embedder`, put the pristine container back so later
    # unpatch cycles restore a clean tree.
    seq = getattr(t_embedder, "_orig", None)
    if isinstance(seq, nn.Module):
        try:
            dm.t_embedder = seq
            t_embedder = seq
        except Exception:
            pass

    rec = {"t1": t1, "norm": norm, "timesteps": t_embedder[0]}
    try:
        _PRISTINE[dm] = rec
    except TypeError:  # non-weakref-able mock in tests
        pass
    return rec


# --------------------------------------------------------------------------
# same-class shallow-copy shims
# --------------------------------------------------------------------------

def _build_temb_shim(pristine, branch):
    """Copy of the pristine TimestepEmbedding whose forward adds the k MLP
    output to the AdaLN-LoRA projection (second return value)."""
    cls_forward = type(pristine).forward
    shim = copy.copy(pristine)

    def forward(self, sample, *args, **kwargs):
        emb, adaln = cls_forward(self, sample, *args, **kwargs)
        if adaln is None:
            raise RuntimeError(
                "Anima TDM-Unify: this model's t_embedder has no AdaLN-LoRA "
                "output; the k branch only supports cosmos-predict2/anima "
                "style models (use_adaln_lora=True)."
            )
        return emb, adaln + branch.k_temb(sample)

    shim.forward = types.MethodType(forward, shim)
    setattr(shim, _STATE_ATTR, branch)
    return shim


def _build_norm_shim(pristine, branch):
    """Copy of the pristine t_embedding_norm whose forward adds the k norm
    output after the original normalization. Because the shim keeps the
    original class, ComfyUI's cast/weight-patch machinery (comfy_cast_weights,
    weight_function lists, lowvram weight streaming, quantized weights) keeps
    working exactly as on the base module."""
    cls_forward = type(pristine).forward
    shim = copy.copy(pristine)

    def forward(self, sample, *args, **kwargs):
        out = cls_forward(self, sample, *args, **kwargs)
        return out + branch.k_embedded(sample)

    shim.forward = types.MethodType(forward, shim)
    setattr(shim, _STATE_ATTR, branch)
    return shim


# --------------------------------------------------------------------------
# live-K plumbing (auto step count from the sampler's sigma schedule)
# --------------------------------------------------------------------------

def _install_live_k_wrapper(patcher, branch):
    """Chain a model_function_wrapper that publishes the sampler's actual
    step count to the k branch (per-thread) before each model call."""
    if not branch.auto_k:
        return
    old = patcher.model_options.get("model_function_wrapper")

    def model_function_wrapper(apply_model, args):
        c = args.get("c") or {}
        to = c.get("transformer_options") or {}
        sigmas = to.get("sample_sigmas")
        if sigmas is not None:
            try:
                branch.set_live_sigmas(sigmas)
            except Exception:
                pass
        try:
            if old is not None:
                return old(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])
        finally:
            branch.clear_live_sigmas()

    patcher.set_model_unet_function_wrapper(model_function_wrapper)


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------

def _get_dm(patcher):
    try:
        return patcher.get_model_object("diffusion_model")
    except AttributeError as exc:
        raise RuntimeError(
            "Input MODEL has no diffusion_model - wire the MODEL output of a "
            "loader/LoRA chain into this node."
        ) from exc


def _validate(pristine, tensors):
    missing = [k for k in PACK_KEYS if k not in tensors]
    if missing:
        raise RuntimeError("extra tensors file is missing tensors: %s" % ", ".join(missing))

    t1, norm = pristine["t1"], pristine["norm"]
    if not isinstance(t1, nn.Module) or not isinstance(norm, nn.Module):
        raise RuntimeError("t_embedder/t_embedding_norm are not modules")
    if not getattr(t1, "use_adaln_lora", False):
        raise RuntimeError(
            "This model's t_embedder does not use AdaLN-LoRA timestep "
            "embeddings - the TDM-Unify k branch only applies to "
            "cosmos-predict2 / anima architecture models."
        )
    l1 = getattr(t1, "linear_1", None)
    l2 = getattr(t1, "linear_2", None)
    norm_w = getattr(norm, "weight", None)
    if l1 is None or l2 is None or norm_w is None:
        raise RuntimeError("t_embedder/t_embedding_norm have an unexpected layout")
    dim = int(norm_w.shape[0])

    expect = {
        "time_embed.k_embed.norm.weight": (dim,),
        "time_embed.k_embed.t_embedder.linear_1.weight": (dim, dim),
        "time_embed.k_embed.t_embedder.linear_2.weight": (3 * dim, dim),
        "time_embed.k_scale": (),
    }
    for key, shape in expect.items():
        got = tuple(tensors[key].shape)
        if got != shape:
            raise RuntimeError(
                "tensor %s has shape %s but the model expects %s - the extra "
                "tensors file does not match this model" % (key, got, shape)
            )


class AnimaTDMUnify:
    """Apply the TDM-Unify k branch to a converted Anima-Lightning model.

    Deterministic across workflow hops / cache evictions / re-executions:
    shims are always rebuilt from the pristine modules, so re-running this
    node is idempotent and can never stack corrections. Bypassing the node
    restores the exact base model behaviour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {
                    "default": DEFAULT_STEPS, "min": 1, "max": 64, "step": 1,
                    "tooltip": "K: sampler step count the TDM-Unify branch conditions on. "
                               "Used as fallback when auto_k is off, or when the sampler's "
                               "sigma schedule can't be read at runtime. Keep it equal to "
                               "your KSampler steps. Official recipe: 4 (CFG 1.0, "
                               "sgm_uniform, shift 3).",
                }),
            },
            "optional": {
                "auto_k": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Derive K at sampling time from the sampler's actual sigma "
                               "schedule (denoise-aware, always in sync). Falls back to "
                               "'steps' when unavailable. Turn off to force the manual "
                               "'steps' value.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "anima"

    DESCRIPTION = (
        "Injects the 4-tensor TDM-Unify k branch (step-count time conditioning) "
        "into converted Anima-Lightning models, with deterministic, idempotent "
        "patching that survives workflow switches and cache evictions."
    )

    def apply(self, model, steps, auto_k=True):
        steps = int(steps)
        auto_k = bool(auto_k)
        patcher = model.clone()
        dm = _get_dm(patcher)

        try:
            dm.t_embedder[1]
        except Exception as exc:
            raise RuntimeError(
                "This diffusion model has no usable t_embedder[1] for the k branch "
                "to attach to - the MODEL input must be a converted Anima / "
                "Cosmos-Predict2 style checkpoint."
            ) from exc

        tensors, source = _resolve_k_tensors()
        pristine = _pristine_record(dm)
        _validate(pristine, tensors)

        branch = _KBranch(tensors, pristine["timesteps"], steps, auto_k)
        temb_shim = _build_temb_shim(pristine["t1"], branch)
        norm_shim = _build_norm_shim(pristine["norm"], branch)

        patcher.add_object_patch("diffusion_model.t_embedder.1", temb_shim)
        patcher.add_object_patch("diffusion_model.t_embedding_norm", norm_shim)
        _install_live_k_wrapper(patcher, branch)

        k_src = ("auto (fallback %d)" % steps) if auto_k else ("%d" % steps)
        print("[anima-tdm-unify] k branch active: K=%s, pack: %s" % (k_src, source))
        return (patcher,)

    def IS_CHANGED(self, model, steps, auto_k=True):
        # Re-execute when the pack file changes on disk (mtime/size) or when
        # the conditioning inputs change. The MODEL input is handled by
        # ComfyUI's own cache signature.
        return (int(steps), bool(auto_k), _pack_fingerprint())


NODE_CLASS_MAPPINGS = {"AnimaTDMUnify": AnimaTDMUnify}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaTDMUnify": "Anima TDM-Unify (k branch)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
