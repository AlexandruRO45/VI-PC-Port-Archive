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
rem No Python on PATH: look for a per-user install (python.org or winget, "Add to PATH" unticked), else offer to install one.
if not defined PY call :find_user_python
if not defined PY call :offer_python
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

rem ---- Helpers (only reached by "call"; the script exits above) ----

rem Each per-user install under %LOCALAPPDATA%\Programs\Python, highest folder name first; each one is
rem version-checked, so the name order does not matter. Needs no PATH refresh.
:find_user_python
for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do call :try_python "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
exit /b 0

:try_python
if defined PY exit /b 0
if not exist %1 exit /b 0
%1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set PY="%~1"
exit /b 0

rem Ask first, then install Python 3.12 for this user only (no admin): winget if present, else the pinned
rem python.org installer checked against its SHA-256 (also used if winget runs but installs nothing). LEONIDA_PY_INSTALL_DRYRUN=1 prints the command and skips it.
rem On "No" or any failure PY stays unset and the caller prints the manual instructions.
:offer_python
echo Python 3.12 or newer was not found, and the build needs it.
choice /c YN /n /t 60 /d N /m "Download and install Python 3.12 for your user account now? (No admin needed) [Y/N] "
if errorlevel 2 exit /b 0
if not errorlevel 1 exit /b 0
where winget >nul 2>nul && goto :py_winget
:py_download
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYSHA=67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
set "PYEXE=%TEMP%\leonida-python-3.12.10-amd64.exe"
if defined LEONIDA_PY_INSTALL_DRYRUN goto :py_dry_download
echo Downloading Python 3.12.10 from python.org ...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:PYURL -OutFile $env:PYEXE; if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:PYEXE).Hash -ine $env:PYSHA) { Remove-Item -LiteralPath $env:PYEXE -Force; exit 3 }"
if errorlevel 1 goto :py_download_failed
echo Installing Python 3.12.10 ...
start "" /wait "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
del /q "%PYEXE%" >nul 2>nul
goto :py_installed

:py_download_failed
echo The download failed or did not match its checksum, so it was not run.
del /q "%PYEXE%" >nul 2>nul
exit /b 0

:py_dry_download
echo [dry run] would download %PYURL%
echo [dry run] would check SHA-256 %PYSHA%
echo [dry run] would run "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
exit /b 0

:py_winget
set "PYWG=winget install --id Python.Python.3.12 -e --scope user --silent --accept-package-agreements --accept-source-agreements"
if defined LEONIDA_PY_INSTALL_DRYRUN goto :py_dry_winget
echo Installing Python 3.12 with winget ...
call %PYWG%
call :find_user_python
if defined PY exit /b 0
echo winget did not install Python; trying the python.org installer instead.
goto :py_download

:py_dry_winget
echo [dry run] would run %PYWG%
goto :py_download

:py_installed
call :find_user_python
exit /b 0

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
