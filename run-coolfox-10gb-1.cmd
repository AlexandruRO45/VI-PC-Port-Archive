@echo off
rem codec-d/cool_fox (f4b48ad), its own Play-VIctory.cmd pareto-residency settings unchanged,
rem only the game folder, a log file and the frame log added.
set "VICTORY_GAME_DIR=C:\Users\Vyto\Downloads\GTA6"
rem 10 GB TEST 1 (one change against cool_fox's own settings): game buffers that shaders read go to the card, as in our v11 (off by default here).
set VICTORY_CARD_WINDOWS_MB=1792
rem Records every pipeline the title makes, so victory warm --used can build them before later runs.
set VICTORY_PIPELINE_BOOK=1
set VICTORY_FRAME_LOG=1
set RUST_LOG=info
if not exist "C:\Users\Vyto\forks\runs" mkdir "C:\Users\Vyto\forks\runs"
call "C:\Users\Vyto\forks\coolfox\Play-VIctory.cmd" pareto-residency > "C:\Users\Vyto\forks\runs\coolfox-10gb-1.log" 2>&1
