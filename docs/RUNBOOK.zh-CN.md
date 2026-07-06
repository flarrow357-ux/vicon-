# 运行手册

以下命令以 PowerShell 为例。路径可以按新电脑实际位置修改。

## 1. 安装依赖

```powershell
cd "E:\vicon gpt\vicon_grey_labeling_toolkit"
pip install -r requirements.txt
```

## 2. 当前推荐：同目录 g 文件流程

从现在开始，推荐所有新实验都使用 `run_active_g_workflow.py`。这个脚本不会把主要结果藏到阶段目录里，而是直接在原实验文件夹中维护一个 `原文件名g.c3d`：

- 第一次自动连点：输入 `2-3.c3d`，输出并覆盖同目录 `2-3g.c3d`。
- 用户手动补推荐帧：直接打开并保存 `2-3g.c3d`。
- 后续自动迭代：继续输入并覆盖同一个 `2-3g.c3d`。
- 中间文件、备份和报告统一放在同目录 `_processing_reports` 中。

第一步：自动连一轮并推荐人工关键帧。

```powershell
python ".\scripts\run_active_g_workflow.py" `
  --input-c3d "E:\vicon gpt\新实验\input.c3d" `
  --model ".\model\1234.mkr" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --mode suggest
```

此时会生成：

```text
E:\vicon gpt\新实验\inputg.c3d
```

推荐帧位置：

```text
E:\vicon gpt\新实验\_processing_reports\active_stage01_suggest\report_suggest_manual_anchor_frames\suggested_manual_anchor_frames.csv
```

第二步：用户在 Nexus 中打开 `inputg.c3d`，按推荐帧人工连完整并保存。

第三步：继续在同一个 `inputg.c3d` 上自动迭代并连接区间前后。

```powershell
python ".\scripts\run_active_g_workflow.py" `
  --input-c3d "E:\vicon gpt\新实验\inputg.c3d" `
  --model ".\model\1234.mkr" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --mode final
```

完成后，最终文件仍然是：

```text
E:\vicon gpt\新实验\inputg.c3d
```

## 3. 旧版阶段目录流程

优先使用一键流程：

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\input.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出" `
  --start-frame 起始帧 `
  --end-frame 结束帧
```

该脚本会自动完成正向、逆向、配套文件复制和最终验证。

不填写 `--final-name` 时，最终文件名自动等于输入 C3D 文件名末尾加 `g`。例如输入 `2-2.c3d`，最终输出为 `2-2g.c3d`。如果手动填写的 `--final-name` 没有以 `g` 结尾，脚本也会自动补上。

## 3.1 旧版人工关键帧协作流程

如果希望先让程序找出“最值得人工补全”的帧，使用下面命令。脚本会先连一轮，再根据剩余缺失人体点、同帧灰点数量、周围连续缺失情况计算推荐帧，然后停止，等待人工处理。

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\input.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出_关键帧推荐" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --suggest-manual-frames `
  --stop-after-suggestion
```

推荐结果位置：

```text
stage02_iter1_reverse_grey\report_suggest_manual_anchor_frames\suggested_manual_anchor_frames.csv
```

用户需要在 Nexus 中打开第一轮结果 C3D，按推荐帧人工连完整并保存。保存后，把这个人工保存后的 C3D 作为新的 `--input-c3d`，继续运行完整流程：

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验输出_关键帧推荐\stage02_iter1_reverse_grey\人工保存后的文件.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出_最终版" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --second-iteration `
  --connect-outside
```

该阶段会连续执行：

- 第一轮正向和逆向灰点连接。
- 第二轮正向和逆向灰点连接。
- 从起始完整帧向前连接区间前段。
- 从结束完整帧向后连接区间后段。
- 全文件验证，确认没有新增点、没有删除点，所有新增人体点都来自同帧灰点。

如需分步运行，可使用下面命令。

```powershell
python ".\scripts\forward_connect_and_short_bridge.py" `
  --c3d "E:\vicon gpt\新实验\input.c3d" `
  --model ".\model\1234.mkr" `
  --output "E:\vicon gpt\新实验输出\FORWARD_GREY.c3d" `
  --report-dir "E:\vicon gpt\新实验输出\report_forward" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --radius 60 `
  --head-radius 25 `
  --max-search 5 `
  --max-gap 0 `
  --forward-max-mean-error 25 `
  --forward-min-margin 30 `
  --lbhd-c7-radius 45 `
  --lbhd-c7-max-mean-error 15
```

`--max-gap 0` 表示不做短缺失补点，只连灰点。

## 4. 逆向只跑一次

```powershell
python ".\scripts\reverse_grey_only_headtight.py" `
  --c3d "E:\vicon gpt\新实验输出\FORWARD_GREY.c3d" `
  --model ".\model\1234.mkr" `
  --output "E:\vicon gpt\新实验输出\FINAL_GREY.c3d" `
  --report-dir "E:\vicon gpt\新实验输出\report_reverse" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --radius 60 `
  --head-radius 25 `
  --max-search 5 `
  --max-mean-error 25 `
  --min-margin 30 `
  --lbhd-c7-radius 45 `
  --lbhd-c7-max-mean-error 15
```

## 5. 注意

脚本本身只写 C3D。为了在 Nexus 中正常打开新 trial，需要把原 trial 的配套文件复制到输出文件夹，并把文件名前缀改成和新 C3D 一致。

使用 `run_full_grey_pipeline.py` 时，这一步会自动完成。

## 6. 最终验证

也可以单独运行验证脚本：

```powershell
python ".\scripts\verify_grey_only_result.py" `
  --original-c3d "E:\vicon gpt\新实验\input.c3d" `
  --final-c3d "E:\vicon gpt\新实验输出\stage02_iter1_reverse_grey\inputg_ITER1.c3d" `
  --model ".\model\1234.mkr" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --report-dir "E:\vicon gpt\新实验输出\verify"
```

注意：`--original-c3d` 指的是本轮脚本处理前的输入基线文件。如果用户已经在 Nexus 中手动改过并保存，应使用手动修改后的 C3D 作为本轮输入基线，而不是更早的原始采集文件。

验证通过必须同时满足：

- 区间外未修改。
- 起始帧完整。
- 结束帧完整。
- 新增人体点全部来自原始同帧灰点。
- 程序生成或插值点数量为 0。
- 每一帧有效点总数和原始 C3D 完全一致，不能增加点，也不能删除点。
