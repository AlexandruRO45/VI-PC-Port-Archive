@echo off
setlocal
set "PACKAGE=C:\Users\paw\Downloads\a"
set "OUT=%~dp0..\..\work\leonida-shader-cache"
set "WORK=%~dp0..\..\work\leonida-shader-work"
set "PATH=C:\Users\paw\Documents\Codex\2026-11-23\yo-x20\work\leonida-shader-tools\env\Library\bin;%PATH%"
cd /d "%~dp0.."
python -u make_game.py --game "%PACKAGE%" --out "%OUT%" --work "%WORK%" --stages shaders --no-tui --wave64-emulate on --jobs 14
echo.
echo Shader extraction finished with exit code %ERRORLEVEL%.
pause
