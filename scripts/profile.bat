@echo off
rem Profiles the engine with Nsight Systems and prints where the GPU time goes.
rem   scripts\profile.bat [model_dir]
rem Needs nsys on PATH (Nsight Systems ships with the CUDA Toolkit; its folder is
rem "C:\Program Files\NVIDIA Corporation\Nsight Systems <version>\target-windows-x64").
rem Reports go to build\prof\ (.nsys-rep opens in the Nsight Systems GUI).
setlocal
set MODEL=%~1
if "%MODEL%"=="" set MODEL=models\qwen2.5-0.5b-instruct
set IE=build\release\ie.exe
set OUT=build\prof

where nsys >nul 2>nul
if errorlevel 1 (
  echo nsys not found. Add the Nsight Systems target-windows-x64 folder to PATH, e.g.
  echo   set PATH=%%PATH%%;C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.1\target-windows-x64
  exit /b 1
)
if not exist %OUT% mkdir %OUT%

call :profile decode512  "bench %MODEL% --prompt-tokens 512 --gen-tokens 256"
call :profile decode4000 "bench %MODEL% --prompt-tokens 4000 --gen-tokens 256"
call :profile serve64    "serve-bench %MODEL% --prompt-tokens 512 --gen-tokens 256 --concurrency 64"

for %%N in (decode512 decode4000 serve64) do (
  echo.
  echo ==================== %%N ====================
  python scripts\profile_summary.py %OUT%\%%N.sqlite
)
exit /b 0

rem --cuda-graph-trace=node: time every kernel inside a CUDA graph, not just the whole graph.
:profile
nsys profile -t cuda,nvtx --cuda-graph-trace=node --export sqlite --force-overwrite true -o "%OUT%\%~1" %IE% %~2
exit /b
