"""Fuse complete regional and scene predictions before Forge applies CFG."""

import math

import torch
from torch.nn.functional import interpolate

from backend.sampling.condition import ConditionConstant, ConditionCrossAttn
from .logging import logger


class RegionalDenoise:
    def __init__(self, fc_args, *, scene_context=None, scene_weight=0.0,
                 fade_start=1.0, region_end=1.0, local_prediction=False, context_percent=5.0):
        if not (0 <= scene_weight <= 1 and 0 <= fade_start <= region_end <= 1):
            raise ValueError("Regional denoise weights/times must be within 0..1, with fade start <= end.")
        self.scene_context = scene_context
        if not 0 <= context_percent <= 25:
            raise ValueError("Local context must be within 0..25 percent.")
        self.local_prediction = local_prediction
        self.context_percent = context_percent
        self.scene_weight = scene_weight
        self.fade_start = fade_start
        self.region_end = region_end
        self._sigma_range = None
        self._phase = None
        count = len(fc_args) // 2
        self.masks = torch.cat([fc_args[f"mask_{i}"] for i in range(1, count + 1)])
        self.contexts = [fc_args[f"cond_{i}"][0] for i in range(1, count + 1)]
        if not count or any(c.ndim != 3 or c.shape[0] != 1 for c in self.contexts):
            raise ValueError("Regional denoise requires one Anima text embedding per region.")
        if not torch.isfinite(self.masks).all() or (self.masks < 0).any():
            raise ValueError("Regional masks must contain finite nonnegative weights.")
        if (self.masks.sum(0) <= 0).any():
            raise ValueError("Regional masks must cover the entire image.")
        self._mask_key = None
        self._latent_masks = None
        self.stats = dict(conditioning_calls=0, conditions=count, scene_weight=scene_weight,
                          fade_start=fade_start, region_end=region_end,
                          regional_calls=0, mixed_calls=0, scene_only_calls=0)
        self.stats.update(local_prediction=local_prediction, context_percent=context_percent)

    def crop_bounds(self, masks):
        height, width = masks.shape[-2:]
        # Align local coordinates to Anima's 2x2 spatial patches. Context affects
        # what the model sees; the original soft mask still controls all writes.
        margin = math.ceil(min(height, width) * self.context_percent / 100)
        bounds = []
        for mask in masks:
            occupied = torch.nonzero(mask > 0)
            if not occupied.numel():
                bounds.append(None)
                continue
            y0, x0 = occupied.amin(0).tolist()
            y1, x1 = (occupied.amax(0) + 1).tolist()
            bounds.append((max(0, (y0-margin)//2*2), min(height, math.ceil((y1+margin)/2)*2),
                           max(0, (x0-margin)//2*2), min(width, math.ceil((x1+margin)/2)*2)))
        return bounds

    def regional_strength(self, model, timestep):
        # Use the model's noise-time conversion, not invocation/step counts:
        # samplers can evaluate the model multiple times for one visible step.
        if self.fade_start == self.region_end == 1.0:
            return 1.0 - self.scene_weight
        if self._sigma_range is None:
            self._sigma_range = (float(model.predictor.percent_to_sigma(self.fade_start)),
                                 float(model.predictor.percent_to_sigma(self.region_end)))
        start, end = self._sigma_range
        sigma = float(timestep.max().item())
        if sigma <= end:
            return 0.0
        if sigma >= start:
            return 1.0 - self.scene_weight
        fraction = (sigma - end) / (start - end)
        return (1.0 - self.scene_weight) * fraction * fraction * (3.0 - 2.0 * fraction)

    def __call__(self, model, cond, uncond, x, timestep, model_options):
        # Forge computes edit_strength before this hook. Replacing an AND prompt
        # here would retain its summed strength and multiply CFG unexpectedly.
        if len(cond) != 1 or cond[0].get("strength", 1.0) != 1.0:
            raise ValueError("Experimental regional denoise does not support AND/composable prompts.")
        if x.ndim not in (4, 5) or (x.ndim == 5 and x.shape[2] != 1):
            raise ValueError("Experimental regional denoise supports image latents only.")
        source = cond[0]
        if any(k in source for k in ("area", "mask", "timestep_start", "timestep_end")):
            raise ValueError("Another extension already changed the conditioning regions or schedule.")
        if self.local_prediction:
            if uncond is not None and len(uncond) != 1:
                raise ValueError("Local regional predictions require a single negative conditioning.")
            for item in [source] + (uncond or []):
                if any(k in item for k in ("control", "area", "mask", "timestep_start", "timestep_end")) or "c_concat" in item["model_conds"]:
                    raise ValueError("Local regional predictions cannot crop ControlNet/inpaint or spatial conditioning.")
        self._regional_strength = self.regional_strength(model, timestep)
        key = (tuple(x.shape[-2:]), x.device)
        if key != self._mask_key:
            masks = interpolate(self.masks[:, None].to(device=x.device, dtype=torch.float32),
                                size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            total = masks.sum(0, keepdim=True)
            if (total <= 0).any():
                raise ValueError("Regional masks leave uncovered latent pixels.")
            self._latent_masks = masks / total
            if self.local_prediction:
                self._crop_bounds = self.crop_bounds(masks)
                self.stats["crop_bounds_yxyx"] = self._crop_bounds
                logger.info(f"[RegionalDenoise] local crops (y0,y1,x0,x1)={self._crop_bounds}; context={self.context_percent:g}%")
            self._mask_key = key
        target_rank = source["model_conds"]["c_crossattn"].cond.ndim
        regional = []
        contexts = list(enumerate(self.contexts)) if self._regional_strength > 0 else []
        if self._regional_strength < 1:
            contexts.append((-1, self.scene_context))
        for index, context in contexts:
            item = source.copy()
            item["model_conds"] = source["model_conds"].copy()
            # A None scene context retains Forge's complete original condition.
            if context is not None:
                if target_rank == 4:
                    context = context.unsqueeze(1)
                elif target_rank != 3:
                    raise ValueError("Unsupported Anima conditioning shape.")
                item["model_conds"]["c_crossattn"] = ConditionCrossAttn(context)
                item["cross_attn"] = context
            # A constant slot separates regional model calls without sending a
            # spatial mask through Forge's four-dimensional area/mask helper.
            item["model_conds"]["fc_regional_slot"] = ConditionConstant(index)
            item["strength"] = 1.0
            regional.append(item)
        self.stats["conditioning_calls"] += 1
        self._active_count = len(regional)
        if self.local_prediction and uncond is not None:
            # CFG must compare predictions from the SAME local view. Reusing a
            # full-frame negative with cropped positives changes its meaning.
            negative = []
            for index, _ in contexts:
                item = uncond[0].copy()
                item["model_conds"] = uncond[0]["model_conds"].copy()
                item["model_conds"]["fc_regional_slot"] = ConditionConstant(index)
                negative.append(item)
            uncond = negative
        phase = "scene_only" if self._regional_strength == 0 else "mixed" if self._regional_strength < 1 else "regional"
        self.stats[phase + "_calls"] += 1
        self.stats["last_regional_strength"] = round(self._regional_strength, 4)
        if phase != self._phase:
            logger.info(f"[RegionalDenoise] phase={phase}; predictions={len(regional)}; "
                        f"regional={self._regional_strength:.3f} scene={1-self._regional_strength:.3f}; "
                        f"fade={self.fade_start:g}..{self.region_end:g}; "
                        f"latent={x.shape[-1]}x{x.shape[-2]}; CFG applied once")
            self._phase = phase
        # The wrapper masks complete predictions; Forge averages these slots.
        # Negative conditioning remains untagged and is evaluated normally.
        return model, regional, uncond, x, timestep, model_options

    def wrap_prediction(self, previous):
        def predict(apply_model, args):
            conditions = args["c"].copy()
            slot = conditions.pop("fc_regional_slot", None)
            call = {**args, "c": conditions}
            crop = None
            if self.local_prediction and slot is not None and slot >= 0:
                crop = self._crop_bounds[slot]
                if crop is None:
                    return torch.zeros_like(args["input"])
                y0, y1, x0, x1 = crop
                call["input"] = args["input"][..., y0:y1, x0:x1].contiguous()
            output = (previous(apply_model, call) if previous is not None else
                      apply_model(call["input"], call["timestep"], **conditions))
            if crop is not None:
                restored = torch.zeros_like(args["input"], dtype=output.dtype)
                restored[..., y0:y1, x0:x1] = output
                output = restored
            if slot is None:
                return output
            if slot == -1:
                return output * ((1.0 - self._regional_strength) * self._active_count)
            mask = self._latent_masks[slot].to(output)
            mask = mask.reshape(*([1] * (output.ndim - 2)), *mask.shape)
            # Forge's unmasked accumulator divides by the number of slots.
            # Compensate here so the result is sum(normalized_mask * prediction),
            # without changing its CFG scale or its negative prediction.
            return output * mask * (self._regional_strength * self._active_count)
        return predict

    @classmethod
    def patch(cls, unet, fc_args, **kwargs):
        hook = cls(fc_args, **kwargs)
        patched = unet.clone()
        previous = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(hook.wrap_prediction(previous))
        patched.set_model_sampler_pre_cfg_function(hook)
        return patched, hook.stats
