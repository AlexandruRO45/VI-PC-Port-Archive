@echo off
setlocal
set "PACKAGE=C:\Users\paw\Downloads\a"
set "REPO=%~dp0.."
set "OUT=%REPO%\..\work\leonida-shader-cache"
set "WORK=%REPO%\..\work\leonida-shader-work"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Prepare Shaders.ps1" -Package "%PACKAGE%" -Repo "%REPO%" -Out "%OUT%" -Work "%WORK%"
if not "%ERRORLEVEL%"=="0" pause
