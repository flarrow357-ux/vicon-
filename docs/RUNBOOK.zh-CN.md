# 运行手册

以下命令以 PowerShell 为例。路径可以按新电脑实际位置修改。

## 1. 安装依赖

```powershell
cd "E:\vicon gpt\vicon_grey_labeling_toolkit"
pip install -r requirements.txt
```

## 2. 正向灰点连接

优先使用一键流程：

```powershell
python ".\scripts\run_full_grey_pipeline.py" `
  --input-c3d "E:\vicon gpt\新实验\input.c3d" `
  --model ".\model\1234.mkr" `
  --output-root "E:\vicon gpt\新实验输出" `
  --start-frame 起始帧 `
  --end-frame 结束帧 `
  --final-name "FINAL_GREY_ONLY"
```

该脚本会自动完成正向、逆向、配套文件复制和最终验证。

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

## 3. 逆向只跑一次

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

## 4. 注意

脚本本身只写 C3D。为了在 Nexus 中正常打开新 trial，需要把原 trial 的配套文件复制到输出文件夹，并把文件名前缀改成和新 C3D 一致。

使用 `run_full_grey_pipeline.py` 时，这一步会自动完成。

## 5. 最终验证

也可以单独运行验证脚本：

```powershell
python ".\scripts\verify_grey_only_result.py" `
  --original-c3d "E:\vicon gpt\新实验\input.c3d" `
  --final-c3d "E:\vicon gpt\新实验输出\stage02_reverse_once_grey\FINAL_GREY_ONLY.c3d" `
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
