@echo off
chcp 65001 >nul
title task2 多智能体视觉识别与科研周报生成
cd /d "%~dp0"

rem 优先使用 Codex 捆绑的 Python，否则使用系统 PATH 中的 python
set "PYTHON_CMD=python"
if exist "C:\Users\34955\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    set "PYTHON_CMD=C:\Users\34955\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

echo ==========================================================
echo   task2 多智能体视觉识别与科研周报生成
echo   模式：视觉大模型（config.json 已配置 API Key）
echo   输入：data\ 目录 8 张图片
echo   输出：logs\ 周报与日志 / figures\ 可视化图
echo ==========================================================
echo.

rem 检查依赖
"%PYTHON_CMD%" -c "import rapidocr_onnxruntime, matplotlib, numpy, PIL" 2>nul
if errorlevel 1 (
    echo [提示] 缺少依赖，正在安装 rapidocr_onnxruntime matplotlib numpy pillow ...
    "%PYTHON_CMD%" -m pip install rapidocr_onnxruntime matplotlib numpy pillow --quiet
)

echo [开始] 运行端到端演示，请稍候...
echo.
"%PYTHON_CMD%" demo_lab_weekly_report.py

echo.
echo ==========================================================
echo   运行结束！结果位置：
echo     周报   : logs\weekly_report.md
echo     日志   : logs\processing_log.json / .txt
echo     评估   : logs\evaluation_report.json
echo     可视化 : figures\swimlane_diagram.png / radar_chart.png / sankey_diagram.png
echo ==========================================================
pause
