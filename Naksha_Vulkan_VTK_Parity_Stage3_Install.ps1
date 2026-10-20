# Naksha LiDAR Stage 3: safe Curve + Freehand integration.
# Default is a read-only dry-run. Requires the completed Stage 1 + Stage 2 worktree.
[CmdletBinding()]
param(
    [ValidateSet('Check', 'Apply', 'Rollback')]
    [string]$Action = 'Check',
    [string]$Project = 'E:\NAKSHA_LIDAR\lidar-classifer-parity-stage1',
    [string]$BundleFolder = $PSScriptRoot,
    [string]$BackupRoot = 'E:\NAKSHA_LIDAR\VTK_VULKAN_REVIEW\Stage3_Backups'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$production = Join-Path $BundleFolder 'Naksha_Vulkan_VTK_Parity_Stage3_Integration.patch'
$tests = Join-Path $BundleFolder 'Naksha_Vulkan_VTK_Parity_Stage3_Tests.patch'
$stage2Hashes = @{
    'gui/native_draw.py' = 'eb05e1b552008fe11ab760a44a522c95278e8a9126c43bfabcf6f0c3dcd8a352'
    'gui/curve_tools.py' = '6b37f0656b5dd6555738a97fe93f0ef531b3dacf26c4c6a1a11825df0a7f4b5e'
}
$stage3Hashes = @{
    'gui/native_draw.py' = '58c238219fb1d78d0b3cd0dcc94b5ea007097025d36b2276d1e93581148f1722'
    'gui/curve_tools.py' = 'ead21ff1706487e2fa55d5133b9b9fb189d871d8e7243dcedce13ac5fd50d5e4'
}
$patchHashes = @{
    'Naksha_Vulkan_VTK_Parity_Stage3_Integration.patch' = 'c0577b4242ca77ea0e6f25c09d671ca545a9a425ae8112527d35503918bdba6c'
    'Naksha_Vulkan_VTK_Parity_Stage3_Tests.patch' = 'c205fdce2bbe0cef4a2606a4f6ff568202537f046863a308df63a287d040b069'
}

function Assert-FileHash([string]$path, [string]$expected, [string]$explanation) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing $explanation : $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "Unexpected $explanation SHA256: $path`nactual: $actual`nexpected: $expected`nStop. Do not force the patch; send this output for a revised patch."
    }
}
function Git-Apply([string]$patch, [switch]$Reverse, [switch]$Check) {
    $argsList = @('-c', 'core.whitespace=cr-at-eol', 'apply')
    if ($Reverse) { $argsList += '-R' }
    if ($Check) { $argsList += '--check' }
    $argsList += $patch
    & git @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Git apply failed for $patch (reverse=$Reverse, check=$Check). Stop."
    }
}
function Source-Hashes([hashtable]$hashes, [string]$description) {
    foreach ($relative in $hashes.Keys) {
        $file = Join-Path $Project ($relative.Replace('/', '\'))
        Assert-FileHash $file $hashes[$relative] $description
    }
}
function Backup-Files {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $folder = Join-Path $BackupRoot $stamp
    if (Test-Path -LiteralPath $folder) {
        throw "Backup folder already exists: $folder. Retry after one second."
    }
    New-Item -Path (Join-Path $folder 'gui') -ItemType Directory -Force | Out-Null
    foreach ($name in @('native_draw.py', 'curve_tools.py')) {
        Copy-Item -LiteralPath (Join-Path $Project "gui\$name") -Destination (Join-Path $folder "gui\$name")
    }
    @(
        "Source project: $Project"
        "Integration patch: $production"
        "Tests patch: $tests"
        "Backup created: $(Get-Date -Format o)"
        'Rollback: use -Action Rollback while Stage 3 files remain unchanged.'
    ) | Set-Content -LiteralPath (Join-Path $folder 'README.txt') -Encoding UTF8
    Write-Host "Stage 2 source backup created: $folder" -ForegroundColor Green
}

foreach ($item in $patchHashes.Keys) {
    Assert-FileHash (Join-Path $BundleFolder $item) $patchHashes[$item] 'Stage 3 patch'
}
if (-not (Test-Path -LiteralPath $Project -PathType Container)) {
    throw "Worktree folder not found: $Project"
}
if (-not (Test-Path -LiteralPath (Join-Path $Project '.git'))) {
    throw 'Not a Git worktree. Stop.'
}
Push-Location $Project
try {
    $branch = (& git branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'feat/vulkan-vtk-parity-stage1') {
        throw "Wrong branch '$branch'. Expected feat/vulkan-vtk-parity-stage1; never patch Sub_master directly."
    }
    Write-Host "Project: $Project"
    Write-Host "Branch: $branch"
    Write-Host 'Existing worktree changes (Stage 1 + Stage 2 are expected):'
    & git status --short
    if ($LASTEXITCODE -ne 0) { throw 'Git status failed' }

    switch ($Action) {
        'Check' {
            Source-Hashes $stage2Hashes 'Stage 2 source baseline'
            if (Test-Path 'gui/native_stroke_geometry.py') { throw 'Stage 3 geometry module already exists; do not reapply.' }
            if (Test-Path 'tests/test_stage3_curve_freehand.py') { throw 'Stage 3 tests already exist; do not reapply.' }
            Git-Apply $production -Check
            Git-Apply $tests -Check
            Write-Host 'DRY RUN PASS: patches fit; no files changed.' -ForegroundColor Green
        }
        'Apply' {
            Source-Hashes $stage2Hashes 'Stage 2 source baseline'
            if (Test-Path 'gui/native_stroke_geometry.py') { throw 'Stage 3 already installed or new file collision. Stop.' }
            if (Test-Path 'tests/test_stage3_curve_freehand.py') { throw 'Stage 3 test file collision. Stop.' }
            Git-Apply $production -Check
            Git-Apply $tests -Check
            $python = Join-Path $Project 'venv\Scripts\python.exe'
            if (-not (Test-Path -LiteralPath $python)) { throw "Missing worktree Python: $python" }
            Backup-Files
            Git-Apply $production
            Git-Apply $tests
            Source-Hashes $stage3Hashes 'installed Stage 3 source'
            & git -c core.whitespace=cr-at-eol diff --check
            if ($LASTEXITCODE -ne 0) { throw 'Git diff whitespace check failed. Do not merge.' }
            & $python -m pytest -q tests/test_stage1_parity.py tests/test_stage2_parity.py tests/test_stage3_curve_freehand.py
            if ($LASTEXITCODE -ne 0) { throw 'Stage 3 tests failed. Leave worktree for diagnosis; do not merge. You can roll back using -Action Rollback.' }
            Write-Host 'STAGE 3 INSTALLED AND AUTOMATED TESTS PASSED. Windows GUI tests still required.' -ForegroundColor Green
        }
        'Rollback' {
            Source-Hashes $stage3Hashes 'installed Stage 3 source'
            Git-Apply $tests -Reverse -Check
            Git-Apply $production -Reverse -Check
            Git-Apply $tests -Reverse
            Git-Apply $production -Reverse
            Source-Hashes $stage2Hashes 'restored Stage 2 source'
            Write-Host 'ROLLBACK PASS: Stage 3 reversed; Stage 1 + Stage 2 source preserved.' -ForegroundColor Green
        }
    }
}
finally {
    Pop-Location
}
