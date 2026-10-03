"""ComfyUI custom node: Anima TDM-Unify (k branch) injection.

Injects the 4 extra TDM-Unify tensors (step-count time conditioning) into
converted Anima-Lightning models. Patches are clone-scoped: bypassing this
node restores the exact base model behaviour.

Extra tensors file (anima_tdm_k_branch.safetensors) is discovered in order:
  1. a copy next to this node
  2. anywhere under ComfyUI/models/diffusion_models/ (or legacy models/unet/)
  3. K_PACK_URL (downloaded once, cached next to this node)

Uninstalling: delete this folder.
"""

import hashlib
import json
import os
import struct
import urllib.request

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

K_PACK_URL = "https://huggingface.co/awdqwdasdg/Anima-Lightning-Comfyui/resolve/main/anima_tdm_k_branch.safetensors"
# SHA-256 pin for K_PACK_URL. If you point K_PACK_URL at your own copy of the
# file, update this hash to match (or set it to "" to skip verification).
K_PACK_SHA256 = "6fae67d2533503ec5dd28230e051203ad955c8a9087d5079d20bcccecc3294c4"

TDM_SHIFT = 3.0      # tdm_unify_shift from the source config
DEFAULT_STEPS = 4    # Anima-Lightning distilled recipe (4 steps, CFG 1.0)

PACK_KEYS = {
    "time_embed.k_embed.norm.weight": (2048,),
    "time_embed.k_embed.t_embedder.linear_1.weight": (2048, 2048),
    "time_embed.k_embed.t_embedder.linear_2.weight": (6144, 2048),
    "time_embed.k_scale": (),
}
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

_MISSING_PACK_MSG = """anima_tdm_k_branch.safetensors not found. Provide the 4-tensor TDM-Unify extra tensors file in one of these ways:
  1. copy it next to this node:           {node_dir}
  2. put it anywhere under:               ComfyUI/models/diffusion_models/
  3. set K_PACK_URL at the top of this file to a direct https URL of the file
The file is produced by convert_anima_to_comfyui.py v1.4+ (or shared by anyone who ran it)."""


# --- extra tensors loading ---

def _read_pack(path):
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


def _download_url_file(url, dest):
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "comfyui-anima-tdm-unify"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if K_PACK_SHA256:
            h = hashlib.sha256()
            with open(tmp, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            if h.hexdigest() != K_PACK_SHA256:
                raise ValueError(
                    "SHA-256 mismatch: got %s, expected %s" % (h.hexdigest(), K_PACK_SHA256)
                )
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _resolve_k_tensors():
    """Return ({key: cpu tensor}, source label) for the 4 k-branch tensors."""
    failures = []
    if os.path.isfile(SIDECAR_PATH):
        candidates = [("sidecar", SIDECAR_PATH, False)]
    else:
        candidates = []
    found = _find_pack_in_model_dirs()
    if found:
        candidates.append(("models dir", found, False))
    if K_PACK_URL:
        candidates.append(("K_PACK_URL", None, True))
    for label, path, is_url in candidates:
        try:
            if is_url:
                if not os.path.isfile(SIDECAR_PATH):
                    _download_url_file(K_PACK_URL, SIDECAR_PATH)
                path, label = SIDECAR_PATH, "K_PACK_URL (cached locally)"
            key = _cache_key(path)
            if key is None:
                continue
            if key not in _K_CACHE:
                _K_CACHE[key] = _read_pack(path)
            return _K_CACHE[key], label
        except Exception as exc:  # try the next supply chain level
            failures.append("  - %s failed: %r" % (label, exc))
    msg = _MISSING_PACK_MSG.format(node_dir=NODE_DIR)
    if failures:
        msg += "\n\nFailure details:\n" + "\n".join(failures)
    raise RuntimeError(msg)


def _validate_pack(tensors):
    missing = [k for k in PACK_KEYS if k not in tensors]
    if missing:
        raise RuntimeError("extra tensors file is missing tensors: %s" % ", ".join(missing))
    for key, shape in PACK_KEYS.items():
        got = tuple(tensors[key].shape)
        if got != shape:
            raise RuntimeError("tensor %s has shape %s, expected %s" % (key, got, shape))


# --- clone-scoped k-branch shims ---

def _k_condition(state, batch, temporal, device):
    """shifted_reciprocal_condition(K) = shift / (K + shift - 1), shape (B, T)."""
    cond = TDM_SHIFT / (float(state.steps) + TDM_SHIFT - 1.0)
    return torch.full((batch, temporal), cond, dtype=torch.float32, device=device)


class _TDMState:
    """Raw k weights + steps; lazily casts to the live device/dtype once each."""

    def __init__(self, tensors, steps):
        self.k_norm = tensors["time_embed.k_embed.norm.weight"]
        self.k_l1 = tensors["time_embed.k_embed.t_embedder.linear_1.weight"]
        self.k_l2 = tensors["time_embed.k_embed.t_embedder.linear_2.weight"]
        self.k_scale = tensors["time_embed.k_scale"].reshape(())
        self.steps = int(steps)
        self.timesteps = None  # the model's own Timesteps module (set at apply)
        self._cast = {}

    def weights(self, device, dtype):
        key = (device, dtype)
        w = self._cast.get(key)
        if w is None:
            w = tuple(t.to(device=device, dtype=dtype)
                      for t in (self.k_norm, self.k_l1, self.k_l2, self.k_scale))
            self._cast[key] = w
        return w

    def k_terms(self, sample):
        """Return (scale*rmsnorm(k_proj), scale*mlp(k_proj)).

        k_proj uses the model's own Timesteps module for frequency parity.
        """
        kp = self.timesteps(
            _k_condition(self, sample.shape[0], sample.shape[1], sample.device)
        ).to(sample.dtype)
        w_norm, l1, l2, scale = self.weights(sample.device, sample.dtype)
        k_embedded = kp * torch.rsqrt(kp.pow(2).mean(-1, keepdim=True) + 1e-6) * w_norm
        k_temb = F.linear(F.silu(F.linear(kp, l1)), l2)
        return scale * k_embedded, scale * k_temb


class _TDMTimestepEmbeddingShim(nn.Module):
    """State-dict-compatible stand-in for t_embedder[1].

    Registers the original linear_1/linear_2 under the same child names so
    state_dict() is unchanged; adds the k adaln-lora correction to the second
    output. The original module is kept unregistered as the compute path.
    """

    def __init__(self, orig, state):
        super().__init__()
        self.linear_1 = orig.linear_1
        self.linear_2 = orig.linear_2
        if hasattr(orig, "activation"):
            self.activation = orig.activation
        object.__setattr__(self, "_orig", orig)  # compute path, NOT re-registered
        self._state = state

    def forward(self, sample, *args, **kwargs):
        emb, adaln = self._orig(sample, *args, **kwargs)
        _, k_temb = self._state.k_terms(sample)
        return emb, adaln + k_temb


class _TDMTEmbedderShim(nn.Module):
    """State-dict-compatible stand-in for the t_embedder Sequential.

    Index 1 is swapped for the k-augmented shim; children keep the original
    numeric names ("0"/"1") so state_dict() and LoRA key matching are unchanged.
    """

    def __init__(self, orig, state):
        super().__init__()
        self.add_module("0", orig[0])
        self.add_module("1", _TDMTimestepEmbeddingShim(orig[1], state))
        object.__setattr__(self, "_orig", orig)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._modules[str(idx)]
        return self._orig[idx]


class _TDMNormShim(nn.Module):
    """State-dict-compatible stand-in for t_embedding_norm (RMSNorm).

    Shares the original weight Parameter and adds the k correction after
    normalization. RMSNorm is computed here (not delegated to the hidden
    original) so ComfyUI's per-module cast flags apply correctly under
    weight streaming / low-VRAM modes.
    """

    def __init__(self, orig, state):
        super().__init__()
        if hasattr(orig, "weight"):
            self.weight = orig.weight  # same Parameter object -> key parity
        self._normalized_shape = tuple(getattr(orig, "normalized_shape", self.weight.shape))
        self._eps = float(getattr(orig, "eps", 1e-6))
        object.__setattr__(self, "_orig", orig)  # provenance only; NOT used in forward
        self._state = state

    def forward(self, sample, *args, **kwargs):
        w = self.weight
        wf = getattr(self, "weight_function", None)
        if wf is not None:  # comfy functional weight patches (LoRA etc.)
            w = wf(w)
        if w.device != sample.device or w.dtype != sample.dtype:
            w = w.to(device=sample.device, dtype=sample.dtype)
        out = F.rms_norm(sample, self._normalized_shape, w, self._eps)
        return out + self._state.k_terms(sample)[0]


# --- node ---

def _get_dm(patcher):
    try:
        return patcher.get_model_object("diffusion_model")
    except AttributeError as exc:
        raise RuntimeError(
            "Input MODEL has no diffusion_model - wire the MODEL output of a "
            "loader/LoRA chain into this node."
        ) from exc


class AnimaTDMUnify:
    """Apply the TDM-Unify k branch to a converted Anima-Lightning model.

    Patches are clone-scoped: applied only when the model flows through this
    node; a bypassed node leaves the model bit-exactly identical to the base.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {
                    "default": DEFAULT_STEPS, "min": 1, "max": 64, "step": 1,
                    "tooltip": "K: sampler step count the TDM-Unify branch conditions on. "
                               "Anima-Lightning official recipe: 4 (with CFG 1.0, sgm_uniform).",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "anima"

    DESCRIPTION = (
        "Injects the 4-tensor TDM-Unify k branch (step-count time conditioning) "
        "into converted Anima-Lightning models. Clone-scoped patches: bypassing "
        "this node restores the exact base model behaviour."
    )

    def apply(self, model, steps):
        steps = int(steps)
        patcher = model.clone()
        dm = _get_dm(patcher)
        tensors, source = _resolve_k_tensors()
        _validate_pack(tensors)
        try:
            orig_tembedder = dm.t_embedder
            orig_norm = dm.t_embedding_norm
        except AttributeError as exc:
            raise RuntimeError(
                "This diffusion model has no t_embedder/t_embedding_norm for the k "
                "branch to attach to - the MODEL input must be a converted "
                "Anima-Lightning checkpoint."
            ) from exc
        state = _TDMState(tensors, steps)
        state.timesteps = orig_tembedder[0]
        patcher.add_object_patch("diffusion_model.t_embedder", _TDMTEmbedderShim(orig_tembedder, state))
        patcher.add_object_patch("diffusion_model.t_embedding_norm", _TDMNormShim(orig_norm, state))
        print(
            "[anima-tdm-unify] k branch armed: K=%d, k_cond=%.6g, weights from %s; "
            "patches are clone-scoped (bypass = off, base model untouched)"
            % (steps, TDM_SHIFT / (steps + TDM_SHIFT - 1.0), source)
        )
        return (patcher,)


NODE_CLASS_MAPPINGS = {"AnimaTDMUnify": AnimaTDMUnify}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaTDMUnify": "Anima TDM-Unify (k branch)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
