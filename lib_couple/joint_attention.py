"""Joint Anima text attention with continuous spatial priors.

Inspired by Sen-sou/Comfyui-Anima-Regional-Conditioning. Unlike hard token
routing, a finite log prior preserves access across region boundaries.
"""
import math
from functools import wraps

import torch
from torch.nn import functional as F

from .attention_masks import get_dit_mask


def spatial_bias(planes, lengths, penalty):
    """Return [1, 1, image tokens, text tokens], retaining soft mask values.

    Length correction gives each prompt a weight budget independent of its
    token count. Zero-weight slots must be removed before this function.
    """
    peak = planes.amax(dim=1, keepdim=True)
    if not math.isfinite(penalty) or not 0 <= penalty <= 12:
        raise ValueError("Joint attention spatial penalty must be between 0 and 12.")
    if len(lengths) != len(planes) or any(n <= 0 for n in lengths):
        raise ValueError("Joint attention context lengths do not match masks.")
    if not torch.isfinite(planes).all() or (planes < 0).any() or (peak <= 0).any():
        raise ValueError("Joint attention requires finite nonnegative, nonempty masks.")
    prior = planes + (peak - planes) * math.exp(-penalty)
    return torch.cat([
        (prior[i].log() - math.log(n))[:, None].expand(-1, n)
        for i, n in enumerate(lengths)
    ], dim=1)[None, None]


class JointAttentionAnima:
    @staticmethod
    @torch.inference_mode()
    def patch_dit(model, width, height, fc_args, penalty=4.0):
        from backend.nn.anima import SelfCrossAttention
        from .anima import AttentionCoupleAnima
        from .logging import logger

        # Share the established cleanup path used before batches and Hi-res.
        AttentionCoupleAnima.unpatch()
        dit = model.model.diffusion_model
        target_ids = {id(m) for m in dit.modules() if isinstance(m, SelfCrossAttention)}
        masks, contexts = [], []
        for i in range(1, len(fc_args) // 2 + 1):
            mask = fc_args[f"mask_{i}"].detach().float()
            if not torch.isfinite(mask).all() or (mask < 0).any():
                raise ValueError("Joint attention received an invalid mask weight.")
            if not mask.any():
                continue
            context = fc_args[f"cond_{i}"][0][0]
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context.ndim != 3 or context.shape[0] != 1:
                raise ValueError("Joint attention expects one text condition per region.")
            masks.append(mask)
            contexts.append(context)
        if not masks:
            raise ValueError("Joint attention has no active regions.")
        masks = torch.stack(masks)
        lengths = [c.shape[1] for c in contexts]
        # Validate the option before installing any model hook.
        if not math.isfinite(penalty) or not 0 <= penalty <= 12:
            raise ValueError("Joint attention spatial penalty must be between 0 and 12.")
        cache = {}
        stats = dict(backend="joint_attention", joint_calls=0, self_attention_bypasses=0,
                     negative_bypasses=0, conditions=len(contexts), text_lengths=lengths,
                     spatial_penalty=penalty)
        original = SelfCrossAttention.forward
        SelfCrossAttention.couple_orig_forward = original

        @wraps(original)
        @torch.inference_mode()
        def forward(self, x, context, rope_emb, transformer_options=None):
            options = transformer_options or {}
            if id(self) not in target_ids or self.is_SelfAttn:
                if id(self) in target_ids:
                    stats["self_attention_bypasses"] += 1
                return original(self, x, context, rope_emb, options)
            flags = options.get("cond_or_uncond")
            if not flags or context is None:
                raise ValueError("Joint attention requires Forge cond_or_uncond metadata.")
            if x.shape[0] % len(flags) or any(f not in (0, 1) for f in flags):
                raise ValueError("Joint attention received unsupported condition batching.")
            outputs = []
            for flag, part, ctx in zip(flags, x.chunk(len(flags)), context.chunk(len(flags))):
                if flag == 1:
                    stats["negative_bypasses"] += 1
                    outputs.append(original(self, part, ctx, rope_emb, options))
                    continue
                key = (part.shape[1], part.device, part.dtype)
                if key not in cache:
                    planes = get_dit_mask(masks.to(part.device), part.shape[1], width,
                                          height, dit.patch_spatial).reshape(len(masks), -1)
                    bias = spatial_bias(planes, lengths, penalty).to(part.dtype)
                    text = torch.cat([c.to(device=part.device, dtype=part.dtype)
                                      for c in contexts], dim=1)
                    cache[key] = (text, bias)
                    stats["mask_mean_weights"] = planes.mean(dim=1).tolist()
                    stats["sequence_length"] = part.shape[1]
                text, bias = cache[key]
                q, k, v = self.compute_qkv(part, text.expand(part.shape[0], -1, -1), rope_emb)
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
                out = out.transpose(1, 2).flatten(2)
                outputs.append(self.output_dropout(self.output_proj(out)))
                stats["joint_calls"] += 1
                if stats["joint_calls"] == 1:
                    logger.info(f"[JointAttention] first_call {stats}")
            return torch.cat(outputs, dim=0)

        forward._couple = True
        SelfCrossAttention.forward = forward
        return model, stats
