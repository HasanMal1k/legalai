# run_app.ps1 -- starts the Pakistan Law Assistant at http://127.0.0.1:8000 (Ctrl+C to stop).
# Starts the Qdrant vector database first if it is not already running (dashboard: http://127.0.0.1:6333/dashboard).
# The API keys are read from your user environment (setx ISAACUS_API_KEY / setx DEEPSEEK_API_KEY).
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
try { Invoke-RestMethod http://127.0.0.1:6333/ -TimeoutSec 2 | Out-Null }
catch {
    New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "data\qdrant") | Out-Null
    Start-Process -FilePath (Join-Path $PSScriptRoot "tools\qdrant\qdrant.exe") `
        -ArgumentList "--config-path `"$(Join-Path $PSScriptRoot 'tools\qdrant\config.yaml')`"" `
        -WorkingDirectory (Join-Path $PSScriptRoot "tools\qdrant") -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $PSScriptRoot "data\qdrant\qdrant.log") -RedirectStandardError (Join-Path $PSScriptRoot "data\qdrant\qdrant.err")
    Start-Sleep -Seconds 5
}
& .\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
