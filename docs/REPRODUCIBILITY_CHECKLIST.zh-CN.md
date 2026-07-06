# 完全复现清单

## 必须包含在仓库中

- `scripts/*.py`
- `model/1234.mkr`
- `model/1234.mp`
- `model/1234.vsk`
- `model/1234.vst`
- `requirements.txt`
- `README.md`
- `docs/*.md`
- `reports/**/*.csv`

## 不得包含在仓库中

- 原始实验 C3D：`*.c3d`
- Nexus 采集数据：`*.x1d`, `*.x2d`, `*.xcp`
- Trial 配套运行文件：`*.system`, `*.history`, `*.Trial.enf`, `*.digitaldevices.xml`
- 视频文件：`*.avi`, `*.mp4`, `*.mov`

这些文件只在本地处理时使用，不上传 GitHub。

## 固定参数

当前最终推荐参数：

- `--radius 60`
- `--head-radius 25`
- `--max-search 5`
- `--max-gap 0`
- `--forward-max-mean-error 25`
- `--forward-min-margin 30`
- `--max-mean-error 25`
- `--min-margin 30`
- `--lbhd-c7-radius 45`
- `--lbhd-c7-max-mean-error 15`
- 人工关键帧推荐：`--manual-top-n 8`
- 人工关键帧窗口：`--manual-window 5`
- 推荐帧最小间隔：`--manual-min-frame-gap 20`

## 推荐执行命令

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "本地实验数据\trial.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "本地输出目录" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --final-name "FINAL_GREY_ONLY"
```

## 推荐的人机协作命令

第一步：自动连一轮并推荐人工关键帧，然后停止。

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "本地实验数据\trial.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "本地输出目录_关键帧推荐" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --final-name "FIRST_PASS_FOR_MANUAL" `
  --suggest-manual-frames `
  --stop-after-suggestion
```

第二步：人工补完推荐帧后，以人工保存后的 C3D 为输入，连续运行两轮迭代并连接区间前后。

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "人工保存后的文件.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "本地输出目录_最终版" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --final-name "FINAL_GREY_ONLY" `
  --second-iteration `
  --connect-outside
```

## 完全复现成功标准

运行结束后，`report_verify/verify_summary.json` 中必须满足。这里的原始文件指本轮脚本处理前的输入基线文件；如果之前有人手动修改过，应以手动修改后的 C3D 作为本轮输入基线。

- `passed` 为 `true`
- `not_from_same_frame_raw` 为 `0`
- `frames_with_point_count_change` 为 `0`
- `changed_before_interval` 为 `false`
- `changed_after_interval` 为 `false`
- `start_complete` 为 `true`
- `end_complete` 为 `true`

如果使用了 `--connect-outside`，还需要检查：

- `stage04_connect_outside/report_verify_full/verify_full_summary.json` 中 `passed` 为 `true`
- `not_from_same_frame_raw` 为 `0`
- `frames_with_point_count_change` 为 `0`
- `new_human_points` 等于 `removed_raw_points`

## 人工抽查重点

即使验证通过，也建议人工抽查：

- `LBHD` 与 `C7` 接近的帧。
- 球拍靠近身体的帧。
- 手部和肘部快速运动帧。
- 遮挡较严重、灰点密集的帧。
