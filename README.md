# Vicon 灰点自动连点工具包

vicon 自动连点羽毛球专项动作数据处理工具。

这是用于 Vicon/Nexus C3D 数据的灰点自动连点脚本集合。当前最终流程原则是：只连接原始数据中已经存在的灰点，不生成新点，不删除点，不做插值补点，不上传实验原始数据。

## 内容

- `scripts/`：全部自动连点、运行和验证脚本。
- `model/`：人体 50 点模型文件，脚本主要读取 `1234.mkr`。
- `reports/`：1-3 实验处理过程中的 CSV 日志和检查报告。
- `docs/WORKFLOW.zh-CN.md`：完整处理流程、规则和质量控制说明。
- `docs/RUNBOOK.zh-CN.md`：运行手册。
- `docs/REPRODUCIBILITY_CHECKLIST.zh-CN.md`：完全复现清单。

## 环境

需要 Python 和 numpy：

```powershell
pip install -r requirements.txt
```

## 一键复现

推荐使用一键流程脚本。它会自动执行：

1. 正向灰点连接。
2. 复制 Nexus trial 配套文件。
3. 逆向只补连一次灰点。
4. 最终验证新增人体点是否全部来自原始同帧灰点。
5. 逐帧验证有效点总数是否完全不变。

如果希望得到更高质量结果，推荐采用“人工关键帧协作流程”：

1. 先自动连一轮，并让脚本推荐最值得人工补全的关键帧。
2. 用户在 Nexus 中打开第一轮结果，把推荐帧人工连完整并保存。
3. 再把人工保存后的 C3D 作为新输入，自动执行第一轮迭代、第二轮迭代和区间前后连接。
4. 最终使用全文件验证，确认没有新增点、没有删除点，所有新人体点都来自同帧灰点。

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\1-3.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出" `
  --start-frame 2301 `
  --end-frame 2889 `
  --final-name "FINAL_GREY_ONLY"
```

人工关键帧推荐阶段：

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\1-3.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出_关键帧推荐" `
  --start-frame 2301 `
  --end-frame 2889 `
  --final-name "FIRST_PASS_FOR_MANUAL" `
  --suggest-manual-frames `
  --stop-after-suggestion
```

推荐帧会写入：

`stage02_iter1_reverse_grey/report_suggest_manual_anchor_frames/suggested_manual_anchor_frames.csv`

人工补完推荐帧并保存后，继续完整自动流程：

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验输出_关键帧推荐\stage02_iter1_reverse_grey\人工保存后的文件.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出_最终版" `
  --start-frame 2301 `
  --end-frame 2889 `
  --final-name "FINAL_GREY_ONLY" `
  --second-iteration `
  --connect-outside
```

如果验证没有通过，脚本会以失败状态退出，并在输出目录写出问题报告。

注意：验证中的原始文件指本轮脚本处理前的输入基线。如果已经有人在 Nexus 中手动改过数据，应使用手动修改后的 C3D 作为本轮输入，而不是更早的采集原始文件。

## 数据边界

仓库不应包含任何实验原始数据或完整轨迹文件，例如：

- `.c3d`
- `.x1d`
- `.x2d`
- `.xcp`
- `.system`
- `.history`
- `.Trial.enf`
- `.digitaldevices.xml`
- 视频文件

这些类型已经写入 `.gitignore`。需要处理新实验时，把实验数据放在本地工作目录，不提交到 GitHub。

## 当前推荐规则

推荐以 `docs/WORKFLOW.zh-CN.md` 为准。核心规则如下：

- 只处理用户指定的起止帧区间。
- 起止帧必须由人工完整连好。
- 只处理人体 50 个 marker。
- 球拍 6 个 marker 永远视为废点，不参与人体连点。
- 只连灰点，不凭空生成点。
- 绝对不能增加或删除点：每一帧有效点总数必须和原始 C3D 完全一致。
- 正向同帧优先，再向后搜索。
- 头部 marker 使用小半径，避免和 C7 混淆。
- 其他人体 marker 保持 60 mm 搜索半径。
- C7 已有效时，LBHD 可以在更严格刚体验证下适当放宽。
- 正向完成后，可只逆向补连一次灰点。
- 第一轮完成后，可推荐“最值得人工补全”的关键帧；人工补帧后再连续执行两轮迭代和区间前后连接。
- 最终必须验证新增点全部来自同帧原始灰点。

## 复现保证

别人下载本仓库后，只要提供本地实验 C3D、对应 trial 配套文件、起止帧，就可以按同一套参数完整复现流程。仓库内不包含实验原始数据；报告日志只保存处理判断信息，不保存完整轨迹数据。最终验证会强制检查“没有增加点、没有删除点、只改变灰点身份”。

## 重要提醒

这套脚本用于辅助科研数据清理。所有自动结果都必须经过人工抽查，尤其是头部、C7、球拍靠近人体、快速挥拍和遮挡严重的帧段。
