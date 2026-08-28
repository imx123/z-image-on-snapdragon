"""Export the fixed-shape Z-Image Transformer as sequential ONNX segments.

Segments:
  frontend      latent/timestep/cap_feats/cap_mask -> unified, freqs, mask, adaln
  layers_NN_MM  unified/freqs/mask/adaln -> unified
  final         unified/adaln -> noise_pred

This keeps each QNN conversion small enough for the QAIRT online quantizer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers import transformer_z_image as _tzi


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


def patch_transformer_for_cap_mask(cls):
    orig_patchify = cls.patchify_and_embed
    orig_prepare = cls._prepare_sequence
    orig_build = cls._build_unified_sequence

    def patchify_and_embed_with_mask(self, all_image, all_cap_feats, patch_size, f_patch_size):
        out = orig_patchify(self, all_image, all_cap_feats, patch_size, f_patch_size)
        all_img_out, all_cap_out, all_img_size, all_img_pos, all_cap_pos, all_img_pad, _ = out
        # Materialize pad masks as FLOAT 0/1 (pad=1) so the ONNX graph stays free
        # of BOOL tensors: HTP's Cast op rejects float<->bool combinations (0xc26).
        cm = self._export_cap_mask.to(torch.float32)
        cap_pad = [(1.0 - cm.clone()) for _ in all_cap_out]
        x_pad = [p.to(torch.float32) for p in all_img_pad]
        self._export_x_pad_mask = x_pad
        self._export_cap_pad_mask = cap_pad
        return (all_img_out, all_cap_out, all_img_size, all_img_pos, all_cap_pos,
                x_pad, cap_pad)

    def prepare_sequence_with_mask(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)
        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        # float 0/1 pad -> lerp to pad_token (no BOOL/Where: HTP rejects float->bool Cast)
        fm = mask.to(feats_cat.dtype)
        feats_cat = feats_cat * (1.0 - fm) + pad_token * fm
        feats = list(feats_cat.split(item_seqlens, dim=0))
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))
        feats = _tzi.pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = _tzi.pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]
        # FLOAT additive attention mask (keep=1.0, pad=0.0) - identical values to
        # the former bool tensor but without BOOL in the ONNX graph (HTP rejects
        # float<->bool Cast). where() below uses the float mask directly.
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.float32, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 0.0
        for i, pad_mask in enumerate(inner_pad_mask):
            n = pad_mask.shape[0]
            pm = pad_mask.to(torch.float32)
            attn_mask[i, :n] = attn_mask[i, :n] * (1.0 - pm)
        if not torch.any(torch.cat(inner_pad_mask)):
            attn_mask = None
        elif len(feats) > 0:
            attn_mask = attn_mask.to(feats[0].dtype)
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = _tzi.pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True, padding_value=0)[:, : feats.shape[1]]
        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    def build_unified_sequence_with_mask(self, x, x_freqs, x_seqlens, x_noise_mask, cap,
                                         cap_freqs, cap_seqlens, cap_noise_mask, siglip,
                                         siglip_freqs, siglip_seqlens, siglip_noise_mask,
                                         omni_mode, device):
        unified, freqs, original_mask, noise = orig_build(
            self, x, x_freqs, x_seqlens, x_noise_mask, cap, cap_freqs, cap_seqlens,
            cap_noise_mask, siglip, siglip_freqs, siglip_seqlens, siglip_noise_mask,
            omni_mode, device)
        if omni_mode:
            return unified, freqs, original_mask, noise
        # Unified mask stays FLOAT (0/1 valid): pads are already float, so slicing
        # and subtraction are pure float - no BOOL Cast anywhere.
        masks = []
        for i in range(len(x_seqlens)):
            x_valid = (1.0 - self._export_x_pad_mask[i])[: x_seqlens[i]]
            cap_valid = (1.0 - self._export_cap_pad_mask[i])[: cap_seqlens[i]]
            masks.append(torch.cat([x_valid, cap_valid], dim=0))
        unified_mask = _tzi.pad_sequence(masks, batch_first=True, padding_value=0.0)
        return unified, freqs, unified_mask, noise

    cls.patchify_and_embed = patchify_and_embed_with_mask
    cls._prepare_sequence = prepare_sequence_with_mask
    cls._build_unified_sequence = build_unified_sequence_with_mask


class FrontendSegment(torch.nn.Module):
    def __init__(self, transformer: ZImageTransformer2DModel):
        super().__init__()
        self.t = transformer

    def forward(self, latent, timestep, cap_feats, cap_mask):
        self.t._export_cap_mask = cap_mask
        x_list = list(latent.unsqueeze(2).unbind(dim=0))
        adaln_input = self.t.t_embedder(timestep * self.t.t_scale).type_as(x_list[0])
        x, cap, x_size, x_pos, cap_pos, x_pad, cap_pad = self.t.patchify_and_embed(
            x_list, [cap_feats], patch_size=2, f_patch_size=1)
        x_seqlens = [len(xi) for xi in x]
        x = self.t.all_x_embedder["2-1"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, _ = self.t._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos, x_pad, self.t.x_pad_token, None, x[0].device)
        for layer in self.t.noise_refiner:
            x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)
        cap_seqlens = [len(ci) for ci in cap]
        cap = self.t.cap_embedder(torch.cat(cap, dim=0))
        cap, cap_freqs, cap_mask_attn, _, _ = self.t._prepare_sequence(
            list(cap.split(cap_seqlens, dim=0)), cap_pos, cap_pad, self.t.cap_pad_token, None, cap[0].device)
        for layer in self.t.context_refiner:
            cap = layer(cap, cap_mask_attn, cap_freqs)
        unified, freqs, mask, _ = self.t._build_unified_sequence(
            x, x_freqs, x_seqlens, None, cap, cap_freqs, cap_seqlens, None,
            None, None, None, None, False, x[0].device)
        return unified, freqs, mask.to(torch.float32), adaln_input


class LayerGroupSegment(torch.nn.Module):
    def __init__(self, transformer: ZImageTransformer2DModel, start: int, end: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([transformer.layers[i] for i in range(start, end)])

    def forward(self, unified, freqs, mask, adaln_input):
        # mask comes in as FLOAT 0/1 (valid); keep it float to avoid a BOOL Cast
        # (HTP rejects float<->bool Cast 0xc26). additively 0/1 == former bool.
        mask = mask.to(torch.float32)
        for layer in self.layers:
            unified = layer(unified, mask, freqs, adaln_input, None, None, None)
        return unified


class FinalSegment(torch.nn.Module):
    def __init__(self, transformer: ZImageTransformer2DModel):
        super().__init__()
        self.t = transformer

    def forward(self, unified, adaln_input):
        out = self.t.all_final_layer["2-1"](unified, c=adaln_input)
        x = self.t.unpatchify(list(out.unbind(dim=0)), [(1, 64, 64)], patch_size=2, f_patch_size=1)
        return torch.stack(x, dim=0).squeeze(2)


def export(module, args, path, names, out_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, args, str(path), input_names=names, output_names=out_names,
        opset_version=17, dynamo=False, do_constant_folding=True, external_data=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out-dir", default="build/transformer_segments_v2")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--layers-per-segment", type=int, default=6)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(args.model_root).resolve()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    transformer = ZImageTransformer2DModel.from_pretrained(
        str(root / "transformer"), torch_dtype=dtype, local_files_only=True).eval()
    patch_transformer_for_cap_mask(ZImageTransformer2DModel)

    out = Path(args.out_dir).resolve()
    latent = torch.zeros((1, 16, 64, 64), dtype=dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    cap = torch.zeros((512, 2560), dtype=dtype)
    cap_mask = torch.zeros((512,), dtype=torch.float32)
    cap_mask[:32] = 1.0

    with torch.no_grad():
        frontend = FrontendSegment(transformer)
        unified, freqs, mask, adaln = frontend(latent, timestep, cap, cap_mask)
        print("intermediate:", tuple(unified.shape), tuple(freqs.shape), tuple(mask.shape), tuple(adaln.shape))
        export(frontend, (latent, timestep, cap, cap_mask), out / "frontend" / "model.onnx",
               ["latent", "timestep", "cap_feats", "cap_mask"],
               ["unified", "unified_freqs", "unified_mask", "adaln_input"])

        manifest = {"groups": [], "shapes": {
            "unified": list(unified.shape), "freqs": list(freqs.shape),
            "mask": list(mask.shape), "adaln": list(adaln.shape)}}
        n_layers = len(transformer.layers)
        for start in range(0, n_layers, args.layers_per_segment):
            end = min(start + args.layers_per_segment, n_layers)
            name = f"layers_{start:02d}_{end - 1:02d}"
            seg = LayerGroupSegment(transformer, start, end)
            ref = seg(unified, freqs, mask, adaln)
            export(seg, (unified, freqs, mask, adaln), out / name / "model.onnx",
                   ["unified_in", "unified_freqs", "unified_mask", "adaln_input"], ["unified_out"])
            manifest["groups"].append({"name": name, "start": start, "end": end})
            unified = ref

        final = FinalSegment(transformer)
        ref = final(unified, adaln)
        print("final:", tuple(ref.shape))
        export(final, (unified, adaln), out / "final" / "model.onnx",
               ["unified_in", "adaln_input"], ["noise_pred"])
        manifest["final"] = "final/model.onnx"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
