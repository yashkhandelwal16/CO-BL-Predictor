$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot "venv\Scripts\python.exe"
$model = Join-Path $projectRoot "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
$expectedHash = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
$spec = Join-Path $projectRoot "packaging\COBLPredictor.spec"
$distDirectory = Join-Path $projectRoot "dist"
$workDirectory = Join-Path $projectRoot "build\pyinstaller"

if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "The Windows application must be built on Windows."
}
if (-not (Test-Path $python -PathType Leaf)) {
    throw "Project virtual-environment Python was not found: $python"
}
if (-not (Test-Path $model -PathType Leaf)) {
    throw "Download the documented Qwen model before building: $model"
}
if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw "The model checksum does not match the verified Qwen model."
}

& $python -m PyInstaller --clean --noconfirm `
    --distpath $distDirectory `
    --workpath $workDirectory `
    $spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

$executable = Join-Path $distDirectory "CO-BL-Predictor.exe"
if (-not (Test-Path $executable -PathType Leaf)) {
    throw "PyInstaller completed without creating the expected executable."
}

$releaseBytes = (Get-Item $executable).Length
$smokeData = Join-Path $env:TEMP (
    "CO-BL-Predictor-smoke-" + [guid]::NewGuid().ToString("N")
)
$previousLocalAppData = $env:LOCALAPPDATA
$previousSelfTestReport = $env:CO_BL_SELF_TEST_REPORT
$previousPath = $env:PATH
$selfTestMarker = Join-Path $distDirectory ".packaged-self-test"
if (Test-Path $selfTestMarker) {
    throw "Remove the existing packaged self-test marker before building: $selfTestMarker"
}
try {
    New-Item -ItemType Directory -Path $smokeData | Out-Null
    $env:LOCALAPPDATA = $smokeData
    $reportPath = Join-Path $smokeData "packaged-self-test.json"
    $env:CO_BL_SELF_TEST_REPORT = $reportPath
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    Set-Content -LiteralPath $selfTestMarker -Value $reportPath -Encoding UTF8
    $smokeProcess = Start-Process `
        -FilePath $executable `
        -ArgumentList "--self-test" `
        -Wait `
        -PassThru
    if ($smokeProcess.ExitCode -ne 0) {
        throw "The packaged model/resource smoke test failed with exit code $($smokeProcess.ExitCode)."
    }
    if (-not (Test-Path $reportPath -PathType Leaf)) {
        throw "The packaged executable did not run its self-test entry point."
    }
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    if (-not $report.frozen -or $report.inference -ne "passed") {
        throw "The packaged smoke test did not verify bundled model inference."
    }

    $userDataDirectory = Join-Path $smokeData "CO-BL-Predictor-MVP"
    if (-not (Test-Path (Join-Path $userDataDirectory "predictor.sqlite3"))) {
        throw "The packaged application did not create its user database."
    }
    if (-not (Test-Path (Join-Path $userDataDirectory "config\bloom_levels.json"))) {
        throw "The packaged application did not initialize writable user configuration."
    }
}
finally {
    $env:LOCALAPPDATA = $previousLocalAppData
    $env:CO_BL_SELF_TEST_REPORT = $previousSelfTestReport
    $env:PATH = $previousPath
    if (Test-Path $selfTestMarker -PathType Leaf) {
        Remove-Item -LiteralPath $selfTestMarker -Force
    }
    if (Test-Path $smokeData -PathType Container) {
        Remove-Item -LiteralPath $smokeData -Recurse -Force
    }
}

Write-Output "Executable: $executable"
Write-Output ("Executable size: {0:N2} GiB" -f ($releaseBytes / 1GB))
Write-Output "Single-file executable model, config, inference, and user-data smoke test: passed"
