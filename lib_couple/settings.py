import gradio as gr

from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts


def fc_settings():
    args = {"section": ("fc", "Forge Couple"), "category_id": "sd"}

    opts.add_option(
        "fc_anima_joint_attention",
        OptionInfo(False, "Mask 使用联合注意力（实验性，仅 Anima）", **args)
        .info("优先于区域去噪融合。全局段描述完整构图与互动，区域段绑定人物；保留整图 self-attention。需选择 First Line 或 Last Line。"),
    )
    opts.add_option(
        "fc_joint_spatial_penalty",
        OptionInfo(4.0, "联合注意力：区域外惩罚", gr.Slider,
                   {"minimum": 0, "maximum": 12, "step": 0.5}, **args)
        .info("0 不限制区域外访问；越高越强调区域位置。始终为柔性约束，不保证人物大小或互动正确。"),
    )

    opts.add_option(
        "fc_mask_regional_denoise",
        OptionInfo(False, "Mask 使用区域去噪融合（实验性，仅 Anima）", **args)
        .info("每个区域运行完整模型预测，按蒙版融合后统一采样；更慢，不保证人物数量或大小。其他模式保持原行为。"),
    )

    opts.add_option(
        "fc_denoise_local_prediction",
        OptionInfo(False, "区域去噪：局部画布预测（实验性）", **args)
        .info("按蒙版范围裁剪 latent，再写回共同画布。每层描述本区域人数（如 1girl），总人数放全局段；暂不支持 ControlNet、局部重绘或参考图。"),
    )
    opts.add_option(
        "fc_denoise_context_percent",
        OptionInfo(5.0, "局部画布：周边上下文 %", gr.Slider,
                   {"minimum": 0, "maximum": 25, "step": 1}, **args)
        .info("按 latent 短边向区域外扩展读取范围；不会扩大写入蒙版。过窄区域可能出现裁切和接缝。"),
    )

    opts.add_option(
        "fc_denoise_scene_weight",
        OptionInfo(0.25, "区域去噪：整图协调强度", gr.Slider,
                   {"minimum": 0, "maximum": 1, "step": 0.05}, **args)
        .info("完整提示词单独预测整幅场景，协调人物关系；0 仅区域，1 仅整图。区别于 Global Effect Weight。"),
    )
    opts.add_option(
        "fc_denoise_fade_start",
        OptionInfo(0.5, "区域去噪：开始渐退的时间", gr.Slider,
                   {"minimum": 0, "maximum": 1, "step": 0.05}, **args)
        .info("按模型噪声时间计（0 开始，1 结束），不是采样步数比例。此后逐渐转为完整场景预测。"),
    )
    opts.add_option(
        "fc_denoise_region_end",
        OptionInfo(0.85, "区域去噪：结束区域约束的时间", gr.Slider,
                   {"minimum": 0, "maximum": 1, "step": 0.05}, **args)
        .info("应不小于渐退时间；之后仅运行完整场景预测。两项时间均为 1 时不渐退。"),
    )

    opts.add_option(
        "fc_do_interrupt",
        OptionInfo(
            True,
            "Interrupt on Error",
            **args,
        )
        .info('if disabled, Forge Couple will simply "fail silently"')
        .needs_restart(),
    )

    opts.add_option(
        "fc_no_presets",
        OptionInfo(
            False,
            "Disable the Presets feature in Advanced mode",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_no_tile",
        OptionInfo(
            False,
            "Disable the Tile mode in img2img",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_adv_newline",
        OptionInfo(
            False,
            "Keep newline characters in Advanced mode dataframe",
            **args,
        ).info('newlines would be shown as "\\n" literals'),
    )


on_ui_settings(fc_settings)
