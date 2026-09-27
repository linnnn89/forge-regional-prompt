# Anima 区域控制探索记录

截至 2026-09-27，后续测试与后端改造暂停。记录已完成的工作，方便恢复研究；不把未验证方案作为产品承诺。

## 冻结范围

- 功能代码基于提交 `b5df75fd2b84f79ed074f1b057fc54a88b3bd650`。本次冻结只补文档，未改变生成逻辑。
- 日常路径：原注意力；Basic / Advanced / Mask 界面保留。
- `fc_anima_joint_attention`、`fc_mask_regional_denoise`、`fc_denoise_local_prediction` 均默认 false；本机保存配置也为 false。旧用户配置不会被强制重置。
- 实验后端仅对 Anima + Mask 生效，联合注意力优先于区域去噪；局部画布依赖区域去噪。
- 不启用实验作为默认，不自动升级 Forge，不下载新模型。不以本记录宣称通用稳定性。

## 目标与验收边界

目标是同一画面中两人占位不等，同时保留自然接触、连续背景、光照和透视。不是独立分格或分开生成人物再拼贴。

蒙版覆盖比例、人物外接框、躯干横向占位、视觉主次是不同指标。头发、裙摆或伸出的手扩大了外接框，不代表人物几何尺度按比例受控。日志证明执行路径，不证明图像合格。

## 本地环境

- Forge Neo：分支 neo，提交 `710f1e25fcac84d880cbccf27d11b2e3276e589e`；生成信息版本 neo-2.29.1。本地宿主可能带有此前诊断修改，不能将提交号当作纯净上游兼容性证明。
- Python 3.13.6，Gradio 4.40.0，PyTorch 2.9.1+cu130，RTX 5070 Ti 16 GB。
- JANIMA_V1，模型 hash `38694ed21a`，qwen_image_vae，qwen_3_06b_base。
- 静态 Anima latent 为 `[B,C,1,H,W]`，不能只允许四维图像 latent。
- 本地 SelfCrossAttention 使用 `is_SelfAttn`；外部报告的 `is_selfattn` 差异不是已确认的本机故障。未进行其他 Forge 版本兼容性验收。

## 已探索的后端

| 路线 | 实现与观察 |
| --- | --- |
| 原注意力 | 各段 cross-attention 输出按蒙版融合，self-attention 保持全局。Basic 已实际使用；非等比人物占位不可靠。 |
| 区域去噪 | 分区域预测完整画面再融合，加入全局预测和按噪声时间退出区域控制；未解决占位与互动同时成立。 |
| 局部画布 | 裁剪区域 latent，正负条件使用相同局部视野；出现背景割裂，保持默认关闭。 |
| 联合注意力 | 拼接各段文本，在一次 cross-attention 的 softmax 前加入连续空间偏置；保留全局 self-attention。能执行，有有限互动改善案例，不能保证比例。 |

早期三图对照使用原注意力 2:1、联合注意力 2:1、联合注意力 1:2。联合组出现手部接触，比例交换主要改变高矮和姿态，横向站位没有可靠反转。不得将这一单 seed 观察称为普遍优于原版。

## A–H 八张后续对照

共同参数：seed `342146220`，1280×1024，25 步，ER SDE / Beta（alpha=0.6，beta=0.6），CFG 5，Shift 3.5，RNG CPU；无 LoRA、无 Hi-res，batch=1。

启用区域控制时：Mask，换行分段，First Line 为完整全局条件，Global Effect Weight=0.5，蒙版重叠与羽化均为 0。关闭区域去噪。空间比例使用硬矩形：2:1 边界 x=853；三等分边界 x=427、853。所有坐标基于 1280×1024。

| ID / 文件名 | 条件变化 | 实图观察 |
| --- | --- | --- |
| A-neutral-off | 中性场景，插件关闭 | 两人全身、尺度接近、手掌接触，背景连续。 |
| B-hierarchy-off | A 加主次描述，插件关闭 | 左侧更大、牵手，但近景裁切下半身。 |
| C-neutral-joint | 中性场景，联合 2:1，penalty=4，区域权重各 1 | 牵手、身份清楚；尺度仍接近，变成近景。 |
| D-hierarchy-joint | C 加主次描述 | 偏背面牵手，左侧略大，仍裁切；未显示超过 B 的比例控制收益。 |
| E-hierarchy-joint-penalty0 | 仅将 D 的 penalty 改为 0 | 偏背面、近景、裁切仍在；单纯取消空间区分没有解决问题。 |
| F-hierarchy-joint-penalty0-region01 | 仅将 E 的两个区域权重各从 1 降到 0.1 | 恢复侧面相对，更接近 B；依旧裁切，并有来源不清的手部。 |
| G-cowboy-2to1 | 近景提示词，2:1，penalty=4，区域权重各 0.1 | 左大右小更明显、牵手；左头顶裁切，不能确认躯干横向 2:1。 |
| H-cowboy-complementary-thirds | 与 G 相同全局条件；三区分别描述 A 左部分 / A 右部分 / B，每区权重 0.1 | 未出现人物复制或明显身体断层；没有明显超过 G，右边缘有可疑手部。 |

这不是严格的多 seed 效果评测。G/H 同时改变了区域数量及局部文本，增加条件也改变了注意力竞争；它们只比较整套使用方案，不隔离“多锚点”机制。G/H 相较 F 还改变了镜头描述和空间惩罚，不能跨组归因于单一因素。

### 可复现提示词

下面每个代码块是一段。A/B 直接用完整全局段；C–H 以换行连接全局段及对应区域段。不要把排版换行误作额外区域。

基础全局段（A/C）：

```text
2girls, anime illustration, best quality. Two adult women face each other and hold hands between them, smiling at each other. Both women are visible in full body on the same beach, summer daylight, ocean and blue sky. The adult woman on the left has long blonde hair and wears a red summer dress. The adult woman on the right has short purple hair and wears a blue summer dress.
```

B/D/E/F 在基础全局段末尾追加：

```text
The blonde woman in the red dress stands slightly closer to the camera and appears larger, her body occupying a broader part of the composition on the left. The purple-haired woman in the blue dress stands slightly farther back and appears smaller on the right. They remain within arm's reach, with their joined hands between them.
```

C/D/E/F 两个区域段：

```text
The adult woman on the left has long blonde hair and wears a red summer dress.
The adult woman on the right has short purple hair and wears a blue summer dress.
```

G/H 使用 D 的全局段，将 `Both women are visible in full body on the same beach,` 替换为 `Close-up, cowboy shot, framing from their heads to mid-thigh on the same beach,`，再追加：

```text
Exactly two adult women are present. The blonde woman is one continuous person extending across the left and middle of the image; the purple-haired woman is on the right.
```

G 两个区域段：

```text
The adult blonde woman in the red summer dress occupies the left and middle of the image. She faces the purple-haired woman and holds her hand.
The adult woman on the right has short purple hair and wears a blue summer dress.
```

H 三个区域段（左右指图像方向，不是人物解剖学左右）：

```text
The image-left portion of the same adult blonde woman in the red summer dress: her long blonde hair and the left side of her torso and dress continue toward the middle of the image.
The image-right portion of that same adult blonde woman in the red summer dress: the right side of her torso and dress continues from the left, and her arm reaches toward the purple-haired woman to hold her hand. This is the same blonde woman.
The adult woman on the right has short purple hair and wears a blue summer dress.
```

全部使用同一负面提示词：

```text
worst quality, low quality, bad anatomy, extra limbs, fused bodies, duplicate, blurry, cropped, text, watermark
```

## 已确认机制与未确认解释

- mask_mapping 分别编码各段；联合路径拼接条件。全局段没有特殊优先权，完整场景中的人物身份也在局部段重复出现。
- 空间先验为 `prior = mask + (peak - mask) * exp(-penalty)`，偏置为 `log(prior) - log(encoded_length)`。penalty=0 时 prior=peak，不再区分位置，但仍有多段条件。
- 本次日志各段编码长度均为 512。长度校正不是按有效单词数或语义信息量分配注意力；0.5:1:1 也不是实际注意力占比。
- F 支持“区域条件权重会影响整体构图”，不证明默认权重是唯一根因。B 已有裁切，因此不能把所有裁切归因于插件。
- 完全相同的 A 条件拆成两个不重叠区域，在原输出融合机制下可等价于合并区域；联合路径还涉及重复文本与区域外泄漏。不能直接宣称复制产生新的空间锚点。
- 三等分互补描述在 H 未拆坏人物，但尚未证明优于 G。close-up 与 cowboy shot 的取景倾向不同，头顶裁切仍存在。

## 已有验证与暂停状态

- 冻结前，联合注意力三项自动检查曾通过：偏置计算、批次与正负向行为、自注意力旁路、恢复与非法权重。自动检查不证明画质。
- A–H 请求均完成并检查原图；联合组日志确认实际进入后端，F 的权重降低与 H 的四段条件均确认。
- 本次文档冻结不再生图，不新增或重跑测试。没有完成多 seed、其他模型、所有扩展、Hi-res、img2img 的完整回归。
- 后续恢复研究须重新明确范围。弱区域权重配合空间限制的独立变量对照尚未完成；G/H 不能替代该对照。不预设下一步自动执行。
- 不下载 ControlNet / LoRA 等权重，不把 BoxDiff、LLLite 或人体部件系统列为已实现能力。

## 本地证据与恢复

原图、请求 JSON、生成信息 JSON 保留在本次 Codex 工作区的 `outputs/composition-factorial`；复现脚本位于同工作区 `work`，文件名分别为 `compare_composition_factorial.py`、`compare_zero_penalty.py`、`compare_scene_dominance.py`、`compare_complementary_thirds.py`。这些个人工作区文件未随仓库发布；本记录提供参数与提示词，不依赖私人绝对路径或在线图片。

日常恢复：关闭联合注意力、区域去噪融合、局部画布预测并 Apply settings。仅将 penalty 设为 0 不会关闭联合注意力。若存在中断后的状态疑问，停止任务并重启 Forge、刷新页面。不要为恢复默认而删除配置或重置整个环境。
