from functools import wraps
from json import dumps, loads

import gradio as gr
import numpy as np
from PIL import Image

from .gr_version import is_neo, js
from .ui_funcs import COLORS
from .mask_layouts import TEMPLATES, create_layout, effective_masks, render_preview

if is_neo:
    from modules_forge.forge_canvas.canvas import ForgeCanvas


class CoupleMaskData:
    def __init__(self, is_img2img: bool):
        self.mode: str = "i2i" if is_img2img else "t2i"
        self.masks: list[Image.Image] = []
        self.weights: list[float] = []
        self.opposite: CoupleMaskData

        self.selected_index: int = -1
        self.overlap = 0.0
        self.feather = 0.0
        self.background = "None"
        self.background_weight = 0.5
        self._effective_key = None
        self._effective_sources = ()
        self._effective_result = []
        self._mask_revision = 0

    def pull_mask(self) -> list[dict]:
        """Pull the masks from the opposite tab"""
        if not self.opposite.masks:
            self.weights = []
            return []

        self.weights = [1.0 for _ in self.opposite.masks]
        # Transfer editable originals; the receiving tab applies its own edge settings.
        return [mask.copy() for mask in self.opposite.masks]

    def _effective_masks(self):
        # Mutations below replace PIL images; retain sources to prevent id reuse.
        key = (tuple(id(mask) for mask in self.masks), self.overlap, self.feather)
        if key != self._effective_key:
            self._effective_result = effective_masks(self.masks, self.overlap, self.feather)
            self._effective_sources = tuple(self.masks)
            self._effective_key = key
        return self._effective_result

    def get_masks(self) -> list[dict]:
        """Return the current masks as well as weights"""
        count = len(self.masks)
        assert count == len(self.weights)

        if count == 0:
            return None

        masks = self._effective_masks()
        return [{"mask": masks[i], "weight": self.weights[i]} for i in range(count)]

    def mask_ui(self, btn, res, mode, background, background_weight) -> list[gr.components.Component]:
        # ===== Components ===== #
        gr.Markdown("**快速开始：选择构图 → 应用模板 → 填写各区域提示词。** 需要不规则形状时，再在下方画布手绘修正。")
        with gr.Group():
            template = gr.Dropdown(TEMPLATES, value=TEMPLATES[0], label="构图模板")
            count = gr.Number(value=2, label="分区数量 N（2–16，整数）", precision=0, minimum=2, maximum=16)
            direction = gr.Radio(["左右", "上下"], value="左右", label="排列方向", visible=False)
            ratio = gr.Textbox(value="2:1", label="比例（例如 2:1、1:2、3:2）", info="双区：左:右或上:下；一大两小：主区:其余区域。", visible=False)
            position = gr.Radio(["左", "右", "上", "下"], value="左", label="主区位置（编号 1）", visible=False)
            replace = gr.Checkbox(False, label="替换已有蒙版图层（保留提示词）")
            apply_template = gr.Button("应用模板并创建图层", variant="primary")
        gr.HTML('<div class="fc_mask_status" role="status" aria-live="polite">尚未创建蒙版图层。</div>')
        with gr.Accordion("交界与融合（0 = 原始硬边）", open=True):
            overlap = gr.Slider(0, 30, value=0, step=0.5, label="重叠宽度 %", info="占画面短边的比例；每层向外扩展此宽度的一半。")
            feather = gr.Slider(0, 30, value=0, step=0.5, label="边缘柔化宽度 %", info="占画面短边的比例；渐变覆盖边界两侧。原始蒙版保留，可随时归零。")
        coverage = gr.Markdown("尚未创建蒙版。")
        gr.Markdown("**有效蒙版预览** · 编号对应提示词；混合色表示共同影响，亮紫色表示未覆盖。")
        msk_preview = gr.Image(show_label=False, image_mode="RGB", type="pil", interactive=False,
                               show_download_button=False, elem_classes="fc_msk_preview", height=320)
        gr.HTML('<h2 align="center"><ins>Mask Layers</ins></h2>')

        gr.HTML('<div class="fc_masks"></div>')

        with gr.Accordion("手绘修正与导入（可选）", open=False):
            gr.Markdown("**手绘修正：** 建立画布 → 用纯白绘制 → 保存为新图层。修改已有图层：选中缩略图 → 载入 → 绘制 → 覆盖选中图层。柔化请使用上方滑块。")
            msk_btn_empty = gr.Button("Create Empty Canvas", elem_classes="round-btn")

            gr.HTML(
                f"""
                <h2 align="center"><ins>Mask Canvas</ins></h2>
                {
                    ""
                    if is_neo
                    else '<p align="center"><b>[Important]</b> Do <b>NOT</b> upload / paste an image to here...</p>'
                }
                """
            )

            msk_canvas = (
                ForgeCanvas(scribble_color="#FFFFFF", no_upload=True)
                if is_neo
                else gr.Image(
                    show_label=False,
                    source="upload",
                    interactive=True,
                    type="pil",
                    tool="color-sketch",
                    image_mode="RGB",
                    brush_color="#ffffff",
                    elem_classes="fc_msk_canvas",
                )
            )

            with gr.Row(elem_classes="fc_msk_io"):
                msk_btn_save = gr.Button(
                    "保存为新图层", interactive=True, elem_classes="round-btn"
                )
                msk_btn_load = gr.Button(
                    "载入选中图层", interactive=False, elem_classes="round-btn"
                )
                msk_btn_override = gr.Button(
                    "覆盖选中图层", interactive=False, elem_classes="round-btn"
                )

            with gr.Row(visible=False):
                operation = gr.Textbox(interactive=True, elem_classes="fc_msk_op")
                operation_btn = gr.Button("op", elem_classes="fc_msk_op_btn")

            msk_gallery = gr.Gallery(
                show_label=False,
                show_share_button=False,
                show_download_button=False,
                interactive=False,
                visible=False,
                elem_classes="fc_msk_gal",
            )

            msk_btn_reset = gr.Button("Reset All Masks", elem_classes="round-btn")

            msk_btn_pull = gr.Button(
                f"Pull from {'txt2img' if self.mode == 'i2i' else 'img2img'}",
                elem_classes="round-btn",
            )

            weights_field = gr.Textbox(visible=False, elem_classes="fc_msk_weights")

            dummy = None if is_neo else gr.State()

            with gr.Row(elem_classes="fc_msk_uploads"):
                upload_background = gr.Image(
                    image_mode="RGBA",
                    label="Upload Background",
                    type="pil",
                    sources="upload",
                    show_download_button=False,
                    interactive=True,
                    height=256,
                    elem_id="fc_msk_upload_bg",
                )

                upload_mask = gr.Image(
                    image_mode="RGBA",
                    label="Upload Mask",
                    type="pil",
                    sources="upload",
                    show_download_button=False,
                    interactive=True,
                    height=256,
                    elem_id="fc_msk_upload_mask",
                )

        # ===== Components ===== #

        # Return preview and status from the same request; programmatic weight
        # outputs do not trigger the user-only input listener below.
        def with_status(fn, single=False, weights=False):
            @wraps(fn)
            def update(*args, **kwargs):
                value = fn(*args, **kwargs)
                result = [value] if single else list(value)
                result.append(self._coverage_status())
                if weights:
                    self._mask_revision += 1
                    result.append(dumps({"revision": self._mask_revision, "weights": self.weights}))
                return result
            return update

        # All mask callbacks share mutable tab state. Neo can serialize them
        # without serializing unrelated extensions or txt2img against img2img.
        events = {"concurrency_id": f"couple_masks_{self.mode}"} if is_neo else {}
        latest = {**events, "trigger_mode": "always_last"} if is_neo else {}

        # ===== Events ===== #
        template.change(
            lambda t: [gr.update(visible=t in TEMPLATES[:2]),
                       gr.update(visible=t == "比例双区"),
                       gr.update(visible=t in ("比例双区", "一大两小")),
                       gr.update(visible=t == "一大两小")],
            template, [count, direction, ratio, position],
        )
        apply_template.click(
            with_status(self._apply_template, weights=True), [res, template, count, direction, ratio, position, replace],
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, replace, coverage, weights_field],
            **events,
        ).success(fn=None, **js(f'() => {{ document.querySelector("#forge_couple_{self.mode} .fc_masks").replaceChildren(); ForgeCouple.populateMasks("{self.mode}"); }}'))
        for control in (overlap, feather):
            control.change(with_status(self._set_edges, single=True), [overlap, feather], [msk_preview, coverage], **latest)
        for control in (background, background_weight):
            control.change(with_status(self._set_background, single=True), [background, background_weight], [msk_preview, coverage], **latest)
        if not is_neo:
            msk_canvas.change(
                fn=None, **js(f'() => {{ ForgeCouple.hideButtons("{self.mode}"); }}')
            )

        msk_btn_empty.click(
            fn=self._create_empty,
            inputs=[res],
            outputs=(
                [msk_canvas.background, msk_canvas.foreground]
                if is_neo
                else [msk_canvas, dummy]
            ),
        )

        msk_btn_pull.click(
            with_status(self._pull_mask, weights=True),
            None,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, coverage, weights_field],
            **events,
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        msk_btn_save.click(
            with_status(self._write_mask, weights=True),
            msk_canvas.foreground if is_neo else msk_canvas,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, coverage, weights_field],
            **events,
        ).success(
            fn=self._create_empty,
            inputs=[res],
            outputs=(
                [msk_canvas.background, msk_canvas.foreground]
                if is_neo
                else [msk_canvas, dummy]
            ),
        ).then(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        msk_btn_override.click(
            with_status(self._override_mask, weights=True),
            msk_canvas.foreground if is_neo else msk_canvas,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, coverage, weights_field],
            **events,
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        msk_btn_load.click(
            self._load_mask, None, msk_canvas.foreground if is_neo else msk_canvas
        )

        msk_btn_reset.click(
            with_status(self._reset_masks, weights=True),
            None,
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, coverage, weights_field],
            **events,
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        weights_field.input(with_status(self._write_weights, single=True), weights_field, [msk_preview, coverage], **latest)

        def apply_operation(op, weights):
            if not self._accept_weights(weights):
                raise gr.Error("图层已变化，请等待更新完成后重试。")
            result = with_status(self._on_operation, weights=True)(op)
            # Clearing the request acknowledges success. On failure Gradio keeps
            # the original value, so the finalizer can unlock without editing prompts.
            return [*result, ""]

        operation_btn.click(
            apply_operation,
            [operation, weights_field],
            [msk_gallery, msk_preview, msk_btn_load, msk_btn_override, coverage, weights_field, operation],
            **events,
        ).then(
            fn=None, inputs=operation,
            **js(f'(op) => {{ document.querySelector("#forge_couple_{self.mode} .fc_msk").dispatchEvent(new CustomEvent("fc-operation-done", {{detail: {{success: op === ""}}}})); if (op === "") ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        btn.click(
            fn=with_status(self._refresh_resolution, weights=True),
            inputs=[res, mode],
            outputs=(
                [msk_gallery, msk_preview, msk_canvas.background, msk_canvas.foreground, coverage, weights_field]
                if is_neo
                else [msk_gallery, msk_preview, msk_canvas, dummy, coverage, weights_field]
            ),
            **events,
        ).success(
            fn=None, **js(f'() => {{ ForgeCouple.populateMasks("{self.mode}"); }}')
        )

        upload_background.upload(
            fn=self._on_up_bg,
            inputs=[res, upload_background],
            outputs=[
                msk_canvas.background if is_neo else msk_canvas,
                upload_background,
            ],
        )

        upload_mask.upload(
            fn=self._on_up_mask,
            inputs=[res, upload_mask],
            outputs=[
                msk_canvas.foreground if is_neo else msk_canvas,
                upload_mask,
            ],
        )
        # ===== Events ===== #

        # ===== Pain ===== #
        for comp in (
            template, count, direction, ratio, position, replace, apply_template,
            overlap, feather, coverage,
            msk_btn_empty,
            msk_btn_pull,
            msk_canvas,
            msk_btn_save,
            msk_btn_load,
            msk_btn_override,
            operation,
            operation_btn,
            msk_preview,
            msk_gallery,
            msk_btn_reset,
            weights_field,
            upload_background,
            upload_mask,
        ):
            comp.do_not_save_to_config = True

        if is_neo:
            msk_canvas.foreground.do_not_save_to_config = True
            msk_canvas.background.do_not_save_to_config = True
        else:
            dummy.do_not_save_to_config = True

    @staticmethod
    def _parse_resolution(resolution: str) -> tuple[int, int]:
        """Convert the resolution from width and height slider"""
        w, h = [int(v) for v in resolution.split("x")]
        while w * h > 1024 * 1024:
            w //= 2
            h //= 2

        return (w, h)

    @staticmethod
    def _create_empty(resolution: str) -> list[Image.Image, None]:
        """Generate a blank black canvas"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        return [Image.new("RGB", (w, h)), None]

    @staticmethod
    def _on_up_bg(resolution: str, image: Image.Image) -> list[Image.Image, bool]:
        """Resize the uploaded image"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        image = image.resize((w, h))

        matt = Image.new("RGBA", (w, h), "black")
        matt.paste(image, (0, 0), image)
        image = matt.convert("RGB")

        array = np.asarray(image, dtype=np.int16)
        array = np.clip(array - 64, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

        return [image, gr.update(value=None)]

    @staticmethod
    def _on_up_mask(resolution: str, image: Image.Image) -> list[Image.Image, bool]:
        """Resize the uploaded image"""
        w, h = CoupleMaskData._parse_resolution(resolution)
        image = image.resize((w, h))

        if is_neo:  # Only keep the pure white Mask
            image_array = np.array(image, dtype=np.uint8)
            white_mask = (image_array[..., :3] == [255, 255, 255]).all(axis=-1)
            image_array[~white_mask] = [0, 0, 0, 0]
            image = Image.fromarray(image_array)

        else:
            matt = Image.new("RGBA", (w, h))
            matt.paste(image, (0, 0), image)
            image = matt.convert("RGB")

        return [image, gr.update(value=None)]

    def _on_operation(self, op: str) -> list[list, Image.Image, bool, bool]:
        """Operations triggered from JavaScript"""
        self.selected_index = -1
        mask_update: bool = True

        # Reorder
        if "=" in op:
            from_id, to_id = [int(v) for v in op.split("=")]
            self.masks[from_id], self.masks[to_id] = (
                self.masks[to_id],
                self.masks[from_id],
            )
            self.weights[from_id], self.weights[to_id] = self.weights[to_id], self.weights[from_id]

        # Delete
        elif "-" in op:
            to_del = int(op.split("-")[1])
            del self.masks[to_del]
            del self.weights[to_del]

        # Select
        else:
            self.selected_index = int(op.strip())
            mask_update = False

        return [
            self.masks if mask_update else gr.skip(),
            self._generate_preview() if mask_update else gr.skip(),
            gr.update(interactive=(self.selected_index >= 0)),
            gr.update(interactive=(self.selected_index >= 0)),
        ]

    def _generate_preview(self) -> Image.Image:
        """Create a preview based on cached masks"""
        if not self.masks:
            return None
        masks = self._effective_masks()
        weights = self.weights if len(self.weights) == len(masks) else [1.0]*len(masks)
        bg = max(0.1, self.background_weight) if self.background != "None" else 0.0
        return render_preview(masks, weights, COLORS, bg)

    def _apply_template(self, resolution, template, count, direction, ratio, position, replace):
        if self.masks and not replace:
            raise gr.Error("已有蒙版。请先勾选“替换已有蒙版图层”，再应用模板。")
        try:
            masks = create_layout(self._parse_resolution(resolution), template, count, direction, ratio, position)
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        self.masks = masks
        self.weights = [1.0]*len(masks)
        self.selected_index = -1
        return [self.masks, self._generate_preview(), gr.update(interactive=False),
                gr.update(interactive=False), False]

    def _set_edges(self, overlap, feather):
        self.overlap, self.feather = float(overlap), float(feather)
        return self._generate_preview()

    def _set_background(self, background, weight):
        self.background, self.background_weight = background, float(weight)
        return self._generate_preview()

    def _coverage_status(self):
        if not self.masks:
            return "尚未保存蒙版图层。应用模板，或绘制后点击“保存为新图层”。"
        masks = self._effective_masks()
        weights = self.weights if len(self.weights) == len(masks) else [1.0]*len(masks)
        total = sum(np.asarray(m.convert("L"), dtype=np.float32)*w for m, w in zip(masks, weights))
        bg = self.background != "None"
        missing = 0.0 if bg else float(np.mean(total <= 0)*100)
        status = "覆盖完整" if missing == 0 else f"⚠ 未覆盖 {missing:.2f}%，请补画或启用 Global Effect"
        return f"已保存 **{len(masks)} 层** · {status} · 重叠 {self.overlap:g}% · 柔化 {self.feather:g}%" + (" · 含全局层" if bg else "")

    def _refresh_resolution(
        self, resolution: str, mode: str
    ) -> list[list, Image.Image, Image.Image, None]:
        """Refresh when width or height is changed"""

        if mode != "Mask":
            return [gr.skip(), gr.skip(), None, None]

        (canvas, _) = self._create_empty(resolution)

        w, h = self._parse_resolution(resolution)

        self.masks = [mask.resize((w, h)) for mask in self.masks]
        preview = self._generate_preview()

        return [self.masks, preview, canvas, None]

    def _reset_masks(self) -> list[list, Image.Image, bool, bool]:
        """Clear everything"""
        self.masks.clear()
        self.weights.clear()
        preview = self._generate_preview()

        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _load_mask(self) -> Image.Image:
        """Load a cached mask to canvas based on index"""
        return self.masks[self.selected_index]

    def _override_mask(
        self, img: None | Image.Image
    ) -> list[list, Image.Image, bool, bool]:
        """Override a cached mask based on index"""
        if img is None:
            self.selected_index = -1
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        assert isinstance(img, Image.Image)

        array = np.asarray(img.convert("L"), dtype=np.uint8)
        mask = np.where(array == 255, 255, 0)
        img = Image.fromarray(mask.astype(np.uint8))

        if not bool(img.getbbox()):
            self.selected_index = -1
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        self.masks[self.selected_index] = img.convert("1")
        self.selected_index = -1

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _write_mask(
        self, img: None | Image.Image
    ) -> list[list, Image.Image, bool, bool]:
        """Save a new mask"""
        if img is None:
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        assert isinstance(img, Image.Image)

        array = np.asarray(img.convert("L"), dtype=np.uint8)
        mask = np.where(array == 255, 255, 0)
        img = Image.fromarray(mask.astype(np.uint8))

        if not bool(img.getbbox()):
            return [
                self.masks,
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]

        self.masks.append(img.convert("1"))
        self.weights.append(1.0)

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _pull_mask(self) -> list[list, Image.Image, bool, bool]:
        """Pull masks from opposite tab"""

        self.masks: list[Image.Image] = self.pull_mask()

        preview = self._generate_preview()
        return [
            self.masks,
            preview,
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]

    def _accept_weights(self, weights: str):
        if weights.lstrip().startswith("{"):
            payload = loads(weights)
            # A delayed input from before a layer operation must not replace
            # the weights belonging to the new layer order.
            if payload["revision"] != self._mask_revision:
                return False
            values = [float(v) for v in payload["weights"]]
        else:
            values = [float(v) for v in weights.split(",")] if weights.strip() else []
        if len(values) != len(self.masks):
            raise gr.Error("图层已变化，请等待图层更新后再调整权重。")
        if any(not np.isfinite(v) or not 0 <= v <= 5 for v in values):
            raise gr.Error("蒙版权重必须是 0–5 的有限数值。")
        self.weights = values
        return True

    def _write_weights(self, weights: str):
        """Cache weights from the current layer order only."""
        if not self._accept_weights(weights):
            return gr.skip()
        return self._generate_preview()
