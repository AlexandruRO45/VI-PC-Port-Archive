@echo off
rem codec-d/cool_fox (f4b48ad), its own Play-VIctory.cmd pareto-residency settings unchanged,
rem only the game folder, a log file and the frame log added.
set "VICTORY_GAME_DIR=C:\Users\Vyto\Downloads\GTA6"
set VICTORY_FRAME_LOG=1
set RUST_LOG=info
if not exist "C:\Users\Vyto\forks\runs" mkdir "C:\Users\Vyto\forks\runs"
call "C:\Users\Vyto\forks\coolfox\Play-VIctory.cmd" pareto-residency > "C:\Users\Vyto\forks\runs\coolfox.log" 2>&1
