# run_enrich.ps1
#
# Runs the second metadata pass (enrich_metadata.py) to completion and
# survives power cuts: a copy of the launcher in the Windows Startup folder
# starts this script at every logon, it carries on from where the last run
# stopped, and it removes that launcher once every document is done.
#
#   Start now:            powershell -ExecutionPolicy Bypass -File run_enrich.ps1
#   Check progress:       .venv\Scripts\python.exe enrich_metadata.py --remaining
#   Stop auto-resuming:   delete "Resume legal metadata.cmd" from the Startup folder
#                         (Win+R, shell:startup)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$log = Join-Path $PSScriptRoot "data\enrich.log"            # what the runner did, and when
$runLog = Join-Path $PSScriptRoot "data\enrich_output.log"  # what the Python script printed
$startupEntry = Join-Path ([Environment]::GetFolderPath("Startup")) "Resume legal metadata.cmd"

function Log($message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $message" | Add-Content -Path $log -Encoding utf8
}

# Only one copy at a time (a second logon, or starting it by hand while it runs).
$mutex = New-Object System.Threading.Mutex($false, "Global\LegalMetadataEnrich")
if (-not $mutex.WaitOne(0)) { exit }

function Get-Remaining {
    # -1 when the count could not be obtained: never mistake a failed check for "nothing left"
    $out = & $python enrich_metadata.py --remaining 2>$null
    if ($LASTEXITCODE -eq 0 -and "$out" -match '^\d+$') { [int]"$out" } else { -1 }
}

function Test-Ollama {
    try { Invoke-WebRequest http://localhost:11434/api/version -UseBasicParsing -TimeoutSec 5 | Out-Null; $true }
    catch { $false }
}

try {
    Log "runner started"

    # The Ollama app may be starting on its own at logon: give it a moment
    # before starting a server ourselves.
    $waited = 0
    while (-not (Test-Ollama) -and $waited -lt 45) { Start-Sleep 5; $waited += 5 }
    if (-not (Test-Ollama)) {
        $env:OLLAMA_NUM_PARALLEL = "4"
        Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
        $waited = 0
        while (-not (Test-Ollama) -and $waited -lt 60) { Start-Sleep 5; $waited += 5 }
    }
    if (-not (Test-Ollama)) { Log "Ollama did not start; giving up for this session"; exit 1 }

    # A pass can leave a few documents undone (a request that failed three
    # times), so go round until nothing is left or no progress is made.
    $left = Get-Remaining
    if ($left -lt 0) { Log "could not count the remaining documents; giving up for this session"; exit 1 }
    for ($round = 1; $left -gt 0 -and $round -le 5; $round++) {
        Log "round $round - $left documents left"
        # Out-File (unlike Add-Content) leaves the file readable while the run is going
        & $python -u enrich_metadata.py --source all 2>$null | Out-File -FilePath $runLog -Append -Encoding utf8
        $now = Get-Remaining
        if ($now -lt 0 -or $now -ge $left) { Log "no progress this round ($now left); stopping"; break }
        $left = $now
    }

    if ($left -eq 0) {
        Log "second pass complete"
        if (Test-Path (Join-Path $PSScriptRoot "finalize_metadata.py")) {
            Log "building final metadata"
            & $python -u finalize_metadata.py 2>&1 | Add-Content -Path $log -Encoding utf8
        }
        if (Test-Path $startupEntry) { Remove-Item $startupEntry -Confirm:$false; Log "removed the Startup launcher" }
        Log "all done"
    }
}
finally {
    $mutex.ReleaseMutex()
}
