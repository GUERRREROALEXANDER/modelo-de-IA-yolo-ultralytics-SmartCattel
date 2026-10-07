# Starts the whole SmartCattle stack on this PC: backend (8000), live camera AI (8090) and frontend (5173).
# Each service opens in its own window; close a window (or press Ctrl+C in it) to stop that service.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1
param(
    [string]$Backend = (Join-Path $PSScriptRoot '..\..\backend de proyecto SmartCattle'),
    [string]$Frontend = (Join-Path $PSScriptRoot '..\..\fronted de proyecto SmartCattle')
)
$ErrorActionPreference = 'Stop'
$ai = Resolve-Path (Join-Path $PSScriptRoot '..')
$Backend = Resolve-Path $Backend
$Frontend = Resolve-Path $Frontend

function Start-Service-Window($title, $dir, $command) {
    Start-Process powershell -WorkingDirectory $dir -ArgumentList '-NoExit', '-Command',
        "`$Host.UI.RawUI.WindowTitle = '$title'; $command"
}

Start-Service-Window 'SmartCattle backend' $Backend '.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000'
Start-Sleep 3
Start-Service-Window 'SmartCattle camara IA' $ai 'python -u live.py'
Start-Service-Window 'SmartCattle frontend' $Frontend 'npm run dev -- --port 5173 --strictPort'
Start-Sleep 8
Start-Process 'http://localhost:5173/live'
