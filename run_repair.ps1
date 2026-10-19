# run_repair.ps1 -- runs the vision-model stage of repair_pages.py until every flagged page is settled, then records
# the results (stage apply). Resumable: every page reading is cached, so after a power cut just start it again.
#
#   powershell -ExecutionPolicy Bypass -File run_repair.ps1
#
# It stops after "apply" on purpose: finalize_metadata.py, the knowledge-graph review (data\kg\status_changes.jsonl,
# audit_kg.py) and chunk_statutes.py are run by hand afterwards, because new text can change what the graph reads.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $PSScriptRoot "data\repair_pages.log"
for ($round = 1; $round -le 20; $round++) {
    "$(Get-Date -Format s) round $round" | Out-File -Append -Encoding utf8 $log
    # (piped through Out-File so the log stays UTF-8; Windows PowerShell's *>> would append UTF-16)
    & .\.venv\Scripts\python.exe repair_pages.py --stage vlm --minutes 240 2>&1 | Out-File -Append -Encoding utf8 $log
    if (Select-String -Path $log -Pattern "vlm: all tiers done" -Quiet) { break }
}
& .\.venv\Scripts\python.exe repair_pages.py --stage apply 2>&1 | Out-File -Append -Encoding utf8 $log
"$(Get-Date -Format s) finished; next: finalize_metadata.py, review data\kg\status_changes.jsonl, audit_kg.py, chunk_statutes.py" |
    Out-File -Append -Encoding utf8 $log
