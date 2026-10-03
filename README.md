# ComfyUI-Anima-Lightning

> **Disclaimer:** This was vibe coded by GLM-5.3-Flash.

ComfyUI custom node that injects the 4 extra TDM-Unify tensors (step-count time conditioning) for Anima-Lightning models.

**Note:** Only works with converted Anima-Lightning checkpoints. Other models (including base Anima) were not trained with this step-conditioning mechanism and will not work correctly.

## Related models

[awdqwdasdg/Anima-Lightning-Comfyui](https://huggingface.co/awdqwdasdg/Anima-Lightning-Comfyui) — converted checkpoints, LoRAs, and the extra tensors file.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/awdqwdasdg/ComfyUI-Anima-Lightning.git
```

Then restart ComfyUI.

## Usage

1. Load a converted Anima-Lightning model (e.g. from the [HF repo](https://huggingface.co/awdqwdasdg/Anima-Lightning-Comfyui)).
2. Wire the MODEL output into the **Anima TDM-Unify (k branch)** node.
3. Set **steps** to `4` (official recipe, CFG 1.0, sgm_uniform) — and use the same step count in your KSampler.

The node automatically finds the extra tensors file (`anima_tdm_k_branch.safetensors`) in this order:

1. A copy next to this node
2. Anywhere under `ComfyUI/models/diffusion_models/` (legacy `models/unet/` is also scanned)
3. Downloaded from the Hugging Face repo above (cached next to this node) — happens automatically on first run if not found locally

Bypassing the node restores the exact base model — no residual state.

## Uninstall

Delete this folder.
