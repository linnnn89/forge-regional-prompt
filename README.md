# Forge Regional Prompt

Forge 区域提示词插件，可为画面不同位置设置独立提示词，适合多角色、分区场景和规则构图。界面中仍显示为 **Forge Couple**。

## 原理

插件在注意力计算中，按区域蒙版和权重混合不同提示词的结果，让每段提示词主要影响指定位置。重叠和柔化用于调节区域之间的过渡；它不是把多张成图拼接起来，也不会自动画出漫画分格线。

## 安装

在 Forge 的 **Extensions → Install from URL** 中填入：

```text
https://github.com/linnnn89/forge-regional-prompt
```

安装后重启 Forge 并刷新页面。若已安装原版 Forge Couple，请先禁用原版，避免同时加载两份插件。

已在 Forge Neo、Gradio 4.40.0 和 Anima 环境中验证；其他组合未重新验证。

## 使用方法

展开 **Forge Couple**，勾选 **Enable**，选择一种模式：

- **Basic**：按提示词分段等分画面。`Horizontal` 为左右分列，`Vertical` 为上下分行。
- **Advanced**：手动设置矩形区域的位置和权重，每个区域对应一段提示词。
- **Mask**：选择模板并点击“应用模板并创建图层”，然后填写每层的提示词。支持 N 列、N 行、四宫格、比例双区（如 2:1、1:2）和一大两小；N 可输入 2–16。

默认每行是一段提示词。例如 Basic 左右两列、Global Effect 设为 `None`：

```text
2girls, red-haired girl, white dress, garden
2girls, blue-haired girl, black dress, garden
```

**Mask 常用操作：**

- **重叠宽度**：增加相邻蒙版共同影响的范围；**边缘柔化**：让边界渐变。两者均按画面短边的百分比计算，设为 0 可恢复原始硬边。
- 预览中的编号对应图层，混合色表示共同影响，亮紫色表示未覆盖区域。
- 上下移动会一起移动蒙版、提示词和权重；删除图层也会删除对应区域提示词，全局提示词会保留。
- 不规则区域可展开“手绘修正与导入”，用纯白绘制后保存为新图层；修改已有图层需先载入，再覆盖。

**通用设置：**

- **Global Effect**：将首段或末段作为全局提示词，其余段数应与区域数一致。
- **Hi-res 时关闭区域控制**：首轮按区域生成，Hi-res 阶段停止区域约束，使用普通提示词继续细化。
- 生成前确认提示词与图层数量匹配，并等待预览更新完成。效果取决于模型对提示词和构图的理解。

## 来源与许可

灵感与代码基础来自 [Haoming02/sd-forge-couple](https://github.com/Haoming02/sd-forge-couple)，其上游实现参考了 [laksjdjf 的 Attention Couple](https://github.com/laksjdjf/cgem156-ComfyUI/tree/main/scripts/attention_couple)。本项目独立维护，采用 [GPL-3.0-or-later](LICENSE) 许可证。

Copyright (C) 2023 laksjdjf · Copyright (C) 2026 Haoming02 · Copyright (C) 2026 linnnn89（修改部分）
