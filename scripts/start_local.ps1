# Starts the whole SmartCattle stack on this PC: backend (8000), live camera AI (8090) and frontend (5173).
# No camera opens by itself: each person connects theirs from the Cameras page.
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

# This PC's address on the farm's network: phones reach the panel through it, and the QR points there.
$route = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue | Sort-Object RouteMetric | Select-Object -First 1
$lan = if ($route) { (Get-NetIPAddress -InterfaceIndex $route.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress }
$origins = 'http://localhost:5173,http://127.0.0.1:5173' + $(if ($lan) { ",http://${lan}:5173" })

function Start-Service-Window($title, $dir, $command) {
    Start-Process powershell -WorkingDirectory $dir -ArgumentList '-NoExit', '-Command',
        "`$Host.UI.RawUI.WindowTitle = '$title'; $command"
}

Start-Service-Window 'SmartCattle backend' $Backend "`$env:ALLOWED_ORIGINS='$origins'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
Start-Sleep 3
Start-Service-Window 'SmartCattle camara IA' $ai 'python -u live.py --host 0.0.0.0'
Start-Service-Window 'SmartCattle frontend' $Frontend 'npm run dev -- --host 0.0.0.0 --port 5173 --strictPort'
Start-Sleep 8
Start-Process 'http://localhost:5173/cameras'
