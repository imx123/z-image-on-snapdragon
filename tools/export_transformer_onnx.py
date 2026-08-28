"""Export the local ModelScope Z-Image-Turbo Transformer (DiT) to ONNX.

Fixed shapes (batch 1, 512x512, 8-step Turbo):
  latent      : [1, 16, 64, 64]  (latent before unsqueeze; F=1 added inside)
  timestep    : [1]              (normalized t in [0,1], scaled by t_scale=1000 inside)
  cap_feats   : [512, 2560]      (Qwen3 hidden states, mask-filtered, no batch dim)

Output:
  noise_pred  : [1, 16, 64, 64]  (matches pipeline's stack+squeeze(2), before the
                                  final negation applied by the scheduler loop)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import ZImageTransformer2DModel

# ---- ONNX-safe pad_sequence ----
# torch.nn.utils.rnn.pad_sequence uses control flow that the TorchScript ONNX
# exporter cannot trace. Replace the module-level binding used inside the
# Z-Image transformer with a statically traceable equivalent (cat + stack).
import diffusers.models.transformers.transformer_z_image as _tzi


def _onnx_safe_pad_sequence(sequences, batch_first: bool = True, padding_value: float = 0.0):
    max_len = max(s.shape[0] for s in sequences)
    padded = []
    for s in sequences:
        if s.shape[0] < max_len:
            pad = torch.full(
                (max_len - s.shape[0],) + tuple(s.shape[1:]),
                padding_value,
                dtype=s.dtype,
                device=s.device,
            )
            s = torch.cat([s, pad], dim=0)
        padded.append(s)
    out = torch.stack(padded, dim=0)
    if not batch_first:
        out = out.transpose(0, 1)
    return out


_tzi.pad_sequence = _onnx_safe_pad_sequence


class TransformerWrapper(torch.nn.Module):
    """Wrap the model to accept plain tensors and emit the pipeline's output shape.

    The official pipeline trims cap_feats to valid prompt tokens before calling
    the transformer. For a fixed-shape mobile graph we keep cap_feats at the
    maximum length (512) and pass an explicit cap_mask so padded positions are
    masked out of attention and replaced with cap_pad_token.
    """

    def __init__(self, transformer: ZImageTransformer2DModel):
        super().__init__()
        self.transformer = transformer
        self.cap_mask = torch.zeros(0)
        self._x_pad_mask = None
        self._cap_pad_mask = None

        # Save original bound methods, then patch instance methods. ONNX
        # tracing follows these Python callables.
        self._orig_patchify_and_embed = transformer.patchify_and_embed
        self._orig_prepare_sequence = transformer._prepare_sequence
        self._orig_build_unified_sequence = transformer._build_unified_sequence
        transformer.patchify_and_embed = self._patchify_and_embed_with_mask
        transformer._prepare_sequence = self._prepare_sequence_with_mask
        transformer._build_unified_sequence = self._build_unified_sequence_with_mask

    def _patchify_and_embed_with_mask(self, all_image, all_cap_feats, patch_size, f_patch_size):
        out = self._orig_patchify_and_embed(all_image, all_cap_feats, patch_size, f_patch_size)
        all_img_out, all_cap_out, all_img_size, all_img_pos_ids, all_cap_pos_ids, all_img_pad_mask, _ = out
        # Replace the multiple-of-32 pad mask with the real prompt mask.
        cap_pad_mask = [~self.cap_mask.to(torch.bool) for _ in all_cap_out]
        self._x_pad_mask = all_img_pad_mask
        self._cap_pad_mask = cap_pad_mask
        return (all_img_out, all_cap_out, all_img_size, all_img_pos_ids,
                all_cap_pos_ids, all_img_pad_mask, cap_pad_mask)

    def _prepare_sequence_with_mask(
        self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None
    ):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        feats_cat = torch.where(mask, pad_token, feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))

        freqs_cis = list(
            self.transformer.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0)
        )
        feats = _tzi.pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = _tzi.pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # Always derive the mask from inner_pad_mask; the original method only
        # built a mask when per-item sequence lengths differed, which silently
        # ignores semantic padding when every item has the same padded length.
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1
        for i, pad_mask in enumerate(inner_pad_mask):
            n = pad_mask.shape[0]
            attn_mask[i, :n] = attn_mask[i, :n] & (~pad_mask)
        if not torch.any(torch.cat(inner_pad_mask)):
            attn_mask = None

        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = _tzi.pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    def _build_unified_sequence_with_mask(
        self, x, x_freqs, x_seqlens, x_noise_mask, cap, cap_freqs, cap_seqlens,
        cap_noise_mask, siglip, siglip_freqs, siglip_seqlens, siglip_noise_mask,
        omni_mode, device,
    ):
        unified, unified_freqs, original_mask, unified_noise_tensor = self._orig_build_unified_sequence(
            x, x_freqs, x_seqlens, x_noise_mask, cap, cap_freqs, cap_seqlens,
            cap_noise_mask, siglip, siglip_freqs, siglip_seqlens, siglip_noise_mask,
            omni_mode, device,
        )
        if omni_mode:
            return unified, unified_freqs, original_mask, unified_noise_tensor

        masks = []
        for i in range(len(x_seqlens)):
            x_valid = ~self._x_pad_mask[i][: x_seqlens[i]]
            cap_valid = ~self._cap_pad_mask[i][: cap_seqlens[i]]
            masks.append(torch.cat([x_valid, cap_valid], dim=0))
        unified_mask = _tzi.pad_sequence(masks, batch_first=True, padding_value=False)
        return unified, unified_freqs, unified_mask, unified_noise_tensor

    def forward(
        self, latent: torch.Tensor, timestep: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor
    ) -> torch.Tensor:
        self.cap_mask = cap_mask.to(torch.bool)
        # latent: (1, C, H, W) -> add F dim -> (1, C, 1, H, W) -> per-batch list
        x_list = list(latent.unsqueeze(2).unbind(dim=0))
        model_out_list = self.transformer(x_list, timestep, [cap_feats], return_dict=False)[0]
        # model_out_list: list of (C, 1, H, W) -> stack -> (1, C, 1, H, W) -> squeeze -> (1, C, H, W)
        return torch.stack(model_out_list, dim=0).squeeze(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, help="path to the ModelScope snapshot root")
    parser.add_argument("--out", default="build/transformer.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16", help="weight/activation precision")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    root = Path(args.model_root).resolve()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(root / "transformer"), torch_dtype=dtype, local_files_only=True
    )
    transformer.eval()
    wrapper = TransformerWrapper(transformer).eval()

    latent = torch.zeros((1, 16, 64, 64), dtype=dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    cap_feats = torch.zeros((512, 2560), dtype=dtype)
    cap_mask = torch.zeros((512,), dtype=torch.bool)
    cap_mask[: 32] = True  # example: 32 valid prompt tokens

    with torch.no_grad():
        ref = wrapper(latent, timestep, cap_feats, cap_mask)
    print(f"reference output shape: {tuple(ref.shape)}")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (latent, timestep, cap_feats, cap_mask),
        str(output),
        input_names=["latent", "timestep", "cap_feats", "cap_mask"],
        output_names=["noise_pred"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
