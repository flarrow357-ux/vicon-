# Vicon 灰点自动连点工具包

vicon 自动连点羽毛球专项动作数据处理工具。

这是用于 Vicon/Nexus C3D 数据的灰点自动连点脚本集合。当前最终流程原则是：只连接原始数据中已经存在的灰点，不生成新点，不做插值补点，不上传实验原始数据。

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

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\1-3.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出" `
  --start-frame 2301 `
  --end-frame 2889 `
  --final-name "FINAL_GREY_ONLY"
```

如果验证没有通过，脚本会以失败状态退出，并在输出目录写出问题报告。

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
- 正向同帧优先，再向后搜索。
- 头部 marker 使用小半径，避免和 C7 混淆。
- 其他人体 marker 保持 60 mm 搜索半径。
- C7 已有效时，LBHD 可以在更严格刚体验证下适当放宽。
- 正向完成后，可只逆向补连一次灰点。
- 最终必须验证新增点全部来自同帧原始灰点。

## 复现保证

别人下载本仓库后，只要提供本地实验 C3D、对应 trial 配套文件、起止帧，就可以按同一套参数完整复现流程。仓库内不包含实验原始数据；报告日志只保存处理判断信息，不保存完整轨迹数据。

## 重要提醒

这套脚本用于辅助科研数据清理。所有自动结果都必须经过人工抽查，尤其是头部、C7、球拍靠近人体、快速挥拍和遮挡严重的帧段。
