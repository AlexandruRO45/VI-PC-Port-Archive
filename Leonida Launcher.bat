@echo off
rem Starts the Leonida Launcher (GUI). Needs Python 3.12+ with tkinter (the python.org installer has it).
cd /d "%~dp0"
set "PYW="
for %%V in (3.14 3.13 3.12) do (
  if not defined PYW (
    py -%%V -c "import tkinter" >nul 2>nul && set "PYW=pyw -%%V"
  )
)
if not defined PYW (
  where pythonw >nul 2>nul && set "PYW=pythonw"
)
if not defined PYW (
  echo Python 3.12 or newer is needed: https://www.python.org/downloads/
  pause
  exit /b 1
)
start "" %PYW% "%~dp0_launcher\launcher.pyw"
