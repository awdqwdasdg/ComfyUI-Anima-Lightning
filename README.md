# ComfyUI-Anima-Lightning

> **Disclaimer:** This was vibe coded by GLM-5.3 and GLM-5.3-Flash.

ComfyUI custom node that injects the 4 extra TDM-Unify tensors (step-count time conditioning) for Anima-Lightning models.

**Note:** Only works with converted Anima-Lightning checkpoints or with LoRAs extracted from it. Other models (including base Anima) were not trained with this step-conditioning mechanism and will not work correctly.

**v2 is a deterministic rewrite** — same node, same inputs, existing workflows keep working. It fixes images changing when switching between workflows (same prompt/seed/size giving different results), which v1 caused by stacking the k-branch patch every time the node re-executed with shims still installed. Bypassing the node still restores the exact base model.

## Related models

[awdqwdasdg/Anima-Lightning-Comfyui](https://huggingface.co/awdqwdasdg/Anima-Lightning-Comfyui) — converted checkpoints, LoRAs, and the extra tensors file.

## Install

    cd ComfyUI/custom_nodes
    git clone https://github.com/awdqwdasdg/ComfyUI-Anima-Lightning.git

Then restart ComfyUI. Updating from v1: `git pull` and restart — no workflow changes needed.

The node automatically finds the extra tensors file (`anima_tdm_k_branch.safetensors`, ~34 MB) in this order:

1. A copy next to this node
2. Anywhere under `ComfyUI/models/diffusion_models/` (legacy `models/unet/` is also scanned)
3. Downloaded from the Hugging Face repo above (cached next to this node) — happens automatically on first run if not found locally

The `ANIMA_TDM_K_PACK_URL` environment variable can point the download at a mirror.

## Usage

1. Load a converted Anima-Lightning model (e.g. from the HF repo above).
2. Wire the MODEL output into the **Anima TDM-Unify (k branch)** node.
3. Official recipe: **4 steps, CFG 1.0, `sgm_uniform`** — and use the same step count in your KSampler.

Inputs:

- **steps** (default `4`) — K the branch conditions on. Only used as a fallback; keep it equal to your KSampler steps.
- **auto_k** (default `true`) — derive K at sampling time from the sampler's actual sigma schedule (denoise-aware, always in sync with your KSampler). Falls back to `steps` when unavailable.

Bypassing the node restores the exact base model — no residual state.

## Uninstall

Delete this folder.
