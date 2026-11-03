@echo off
title SmartCattle
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_farm.ps1" %*
pause
