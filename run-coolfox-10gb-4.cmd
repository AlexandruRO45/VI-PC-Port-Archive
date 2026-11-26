@echo off
rem codec-d/cool_fox (f4b48ad), its own Play-VIctory.cmd pareto-residency settings unchanged,
rem only the game folder, a log file and the frame log added.
set "VICTORY_GAME_DIR=C:\Users\Vyto\Downloads\GTA6"
rem 10 GB TEST 2 (one change against cool_fox's own settings; test 1's CARD_WINDOWS_MB=1792 dropped: worse, this fork reads no windows).
rem Card copies of the title's buffers: this fork caps them at 512 MB on cards of 12 GB or less; v11 ran well with 1024.
set VICTORY_CARD_BUFFERS_MB=1024
rem 10 GB TEST 4 (one change against test 2, the failsafe): 64-lane programs run as two host waves with group barriers,
rem so the 5 compute programs dropped in play translate (they did in v11, where the water showed). 26 -> 6 untranslated.
rem Risk noted by the fork: lost card on AMD with ptfx_CS_lightAtlas_Sprite/_Trail.
set VICTORY_WAVE_WIDE=1
rem Records every pipeline the title makes, so victory warm --used can build them before later runs.
set VICTORY_PIPELINE_BOOK=1
set VICTORY_FRAME_LOG=1
set RUST_LOG=info
if not exist "C:\Users\Vyto\forks\runs" mkdir "C:\Users\Vyto\forks\runs"
call "C:\Users\Vyto\forks\coolfox\Play-VIctory.cmd" pareto-residency > "C:\Users\Vyto\forks\runs\coolfox-10gb-4.log" 2>&1
