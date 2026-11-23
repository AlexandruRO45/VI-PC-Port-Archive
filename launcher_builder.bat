@echo off
setlocal
rem Builds Leonida Launcher (Leonida.exe) next to this file. Double-click it; no command prompt needed.
rem Same as: powershell -ExecutionPolicy Bypass -File launcher-wpf\build.ps1
cd /d "%~dp0"
title Leonida Launcher builder

echo.
echo   Leonida Launcher builder
echo   ------------------------
echo   Builds Leonida.exe in this folder, next to Build.bat.
echo.

if not exist "launcher-wpf\build.ps1" (
    echo   launcher-wpf\build.ps1 was not found. Put this file in your Leonida folder, next to Build.bat.
    goto :fail
)

where dotnet >nul 2>nul
if errorlevel 1 goto :nodotnet
dotnet --list-sdks 2>nul | findstr /r "^[89]\. ^1[0-9]\." >nul
if errorlevel 1 goto :nodotnet

tasklist /fi "imagename eq Leonida.exe" 2>nul | find /i "Leonida.exe" >nul
if not errorlevel 1 (
    echo   Leonida Launcher is open. Close it first, then press a key to go on.
    pause >nul
)

echo   Building, this takes a minute the first time...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "launcher-wpf\build.ps1"
if errorlevel 1 goto :fail

echo.
echo   Done. Start Leonida.exe in this folder.
echo.
pause
exit /b 0

:nodotnet
echo   The .NET 8 SDK (or newer) is needed and was not found.
echo   Get it from https://dotnet.microsoft.com/download/dotnet/8.0 (SDK, Windows x64),
echo   install it, then run this file again.
goto :fail

:fail
echo.
echo   Leonida.exe was not built. See the messages above.
echo.
pause
exit /b 1
