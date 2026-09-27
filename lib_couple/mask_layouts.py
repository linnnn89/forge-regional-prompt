"""Mask templates and reversible edge adjustments; coordinates cover all pixels."""

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy.ndimage import distance_transform_edt

TEMPLATES = ["左右 N 列", "上下 N 行", "四宫格", "比例双区", "一大两小"]


def create_layout(size, template, count=2, direction="左右", ratio="2:1", position="左"):
    w, h = size
    boxes = []
    if template in TEMPLATES[:2]:
        n = float(count)
        if not np.isfinite(n) or n != int(n) or not 2 <= n <= 16:
            raise ValueError("分区数量 N 必须是 2–16 的整数；区域越多，显存与计算开销越大。")
        n = int(n)
        horizontal = template == TEMPLATES[0]
        extent = w if horizontal else h
        if n > extent:
            raise ValueError("分区数量不能超过对应方向的像素数。")
        edges = [round(i * extent / n) for i in range(n + 1)]
        boxes = [(edges[i], 0, edges[i+1], h) if horizontal else
                 (0, edges[i], w, edges[i+1]) for i in range(n)]
    elif template == "四宫格":
        boxes = [(x1, y1, x2, y2) for y1, y2 in [(0, h//2), (h//2, h)]
                 for x1, x2 in [(0, w//2), (w//2, w)]]
    elif template in ("比例双区", "一大两小"):
        try:
            a, b = [float(v.strip()) for v in ratio.split(":")]
            if not np.isfinite([a, b, a+b]).all() or min(a, b) <= 0:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValueError("比例请填写两个正数，例如 2:1、1:2 或 3:2。") from None
        horizontal = direction == "左右" if template == "比例双区" else position in ("左", "右")
        extent = w if horizontal else h
        length = round(extent * a / (a+b))
        if not 0 < length < extent:
            raise ValueError("比例过于悬殊，会产生空区域。")
        cut = extent-length if template == "一大两小" and position in ("右", "下") else length
        boxes = [(0, 0, cut, h), (cut, 0, w, h)] if horizontal else [(0, 0, w, cut), (0, cut, w, h)]
        if template == "一大两小":
            if position in ("右", "下"):
                boxes.reverse()
            main, (x1, y1, x2, y2) = boxes
            boxes = [main, (x1, y1, x2, h//2), (x1, h//2, x2, y2)] if horizontal else [main, (x1, y1, w//2, y2), (w//2, y1, x2, y2)]
    else:
        raise ValueError("未知构图模板。")
    masks = []
    for x1, y1, x2, y2 in boxes:
        array = np.zeros((h, w), dtype=np.uint8)
        array[y1:y2, x1:x2] = 255
        masks.append(Image.fromarray(array).convert("1"))
    return masks


def effective_masks(masks, overlap=0.0, feather=0.0):
    """Overlap is total shared width; feather is the full linear transition band.

    Both use % of the shorter image side. Original binary masks are never changed.
    """
    if not 0 <= overlap <= 30 or not 0 <= feather <= 30:
        raise ValueError("重叠与柔化范围必须为 0–30%。")
    if not masks or (overlap == 0 and feather == 0):
        return masks
    result = []
    for mask in masks:
        binary = np.asarray(mask.convert("L")) > 127
        if not binary.any() or binary.all():
            result.append(mask.convert("L"))
            continue
        scale = min(mask.size) / 100.0
        inside = distance_transform_edt(binary)
        outside = distance_transform_edt(~binary)
        signed = np.where(binary, inside-0.5, 0.5-outside)
        shifted = signed + overlap * scale / 2.0
        strength = np.clip(0.5 + shifted / (feather*scale), 0, 1) if feather else (shifted >= 0)
        result.append(Image.fromarray(np.rint(strength*255).astype(np.uint8)))
    return result


def render_preview(masks, weights, colors, background_weight=0.0):
    if not masks:
        return None
    values = np.stack([np.asarray(m.convert("L"), dtype=np.float32)/255 for m in masks])
    values *= np.asarray(weights, dtype=np.float32)[:, None, None]
    total = values.sum(axis=0) + background_weight
    palette = np.array([ImageColor.getrgb(colors[i % len(colors)]) for i in range(len(masks))])
    rgb = np.einsum("nhw,nc->hwc", values, palette) + background_weight * 100
    rgb /= np.maximum(total[..., None], 1e-8)
    # Magenta identifies uncovered pixels; mixed colors show shared influence.
    rgb[total <= 0] = (255, 0, 255)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    for i, value in enumerate(values):
        ys, xs = np.where(value > 0)
        if xs.size:
            x, y = int(xs.mean()), int(ys.mean())
            label = Image.new("RGB", (24, 20), "black")
            ImageDraw.Draw(label).text((5, 4), str(i+1), fill="white")
            scale = max(1.0, min(image.size)/320)
            label = label.resize((round(24*scale), round(20*scale)), Image.Resampling.NEAREST)
            image.paste(label, (x-label.width//2, y-label.height//2))
    return image
