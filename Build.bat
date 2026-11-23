@echo off
title Leonida - Game Builder
rem Double-click to build. Finds your copy of the game, checks what's installed, then asks where to build
rem (Enter keeps Documents\Leonida, or the folder you chose last time). Extra options pass through, e.g.
rem   Build.bat --out D:\Games\Leonida      (no question: --out, --find, --check and --help skip it)
setlocal
cd /d "%~dp0"
echo.
echo  ==========================================
echo       L E O N I D A  -  Game Builder
echo  ==========================================
echo.
set "PY="
for %%V in (3.14 3.13 3.12 3) do (
  if not defined PY (
    py -%%V -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  where python >nul 2>nul && (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY=python"
  )
)
if not defined PY (
  echo Python 3.12 or newer is needed: https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" during setup, then run Build.bat again.
  pause
  exit /b 1
)
set "OUTARG="
echo(%*| findstr /i /r /c:"--out\>" /c:"--find\>" /c:"--check\>" /c:"--help\>" /c:"^-h\>" /c:" -h\>" >nul || call :choose_out
%PY% -I "%~dp0make_game.py" %OUTARG% %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Done.) else (echo The build stopped. Read the message above.)
pause
exit /b %RC%

rem Asks for the build folder and remembers it, so a stopped build resumes in the same place.
:choose_out
set "SAVED=%LOCALAPPDATA%\Leonida\build_out.txt"
set "DEF=%USERPROFILE%\Documents\Leonida"
if exist "%SAVED%" set /p DEF=<"%SAVED%"
echo  Free space on your drives:
powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free } | ForEach-Object { '    {0}:  {1,6:N0} GB free' -f $_.Name, ($_.Free / 1GB) }"
echo.
echo  A full build needs about 400 GB; without the assets stage
echo  (Build.bat --stages check,shaders,runtime) about 30 GB.
echo.
echo  Where should the build go? Type a folder such as D:\Games\Leonida,
echo  or just a drive letter such as D (it becomes D:\Leonida).
echo  Press Enter to use: %DEF%
set "OUTDIR="
set /p "OUTDIR=  Folder: "
if not defined OUTDIR set "OUTDIR=%DEF%"
set "OUTDIR=%OUTDIR:"=%"
if "%OUTDIR:~1%"=="" set "OUTDIR=%OUTDIR%:\Leonida"
if "%OUTDIR:~1%"==":" set "OUTDIR=%OUTDIR%\Leonida"
if "%OUTDIR:~1%"==":\" set "OUTDIR=%OUTDIR%Leonida"
if not exist "%LOCALAPPDATA%\Leonida" mkdir "%LOCALAPPDATA%\Leonida" >nul 2>nul
rem No ( ) block here: a ")" in the folder name, as in "Program Files (x86)", would close it early.
>"%SAVED%" echo(%OUTDIR%
set "OUTARG=--out "%OUTDIR%""
echo.
exit /b 0
