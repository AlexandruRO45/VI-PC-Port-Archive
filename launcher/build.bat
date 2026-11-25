@echo off
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /win32icon:"%~dp0icon.ico" /out:"%~dp0GTA VI Launcher.exe" "%~dp0Launcher.cs"
