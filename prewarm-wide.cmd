@echo off
rem Prewarm shaders and pipelines of the target-wide build for one VICTORY_WAVE_WIDE_ONLY list, then exit.
rem Pass the list in quotes; cmd splits unquoted arguments at commas: prewarm-wide.cmd "hash1,hash2"
set "VICTORY_CACHE_DIR=%LOCALAPPDATA%\VIctory\cache-pareto"
set "VICTORY_WAVE_WIDE_ONLY=%~1"
set "NAME=%~1"
set "NAME=%NAME:,=_%"
cd /d C:\Users\Vyto\forks\coolfox
target-wide\release\victory.exe shaders "C:\Users\Vyto\Downloads\GTA6" > "C:\Users\Vyto\forks\runs\prewarm-%NAME%.log" 2>&1
target-wide\release\victory.exe warm --used >> "C:\Users\Vyto\forks\runs\prewarm-%NAME%.log" 2>&1
exit /b %errorlevel%
