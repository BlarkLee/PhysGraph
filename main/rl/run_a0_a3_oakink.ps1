param(
    [ValidateSet("smoke", "gate")]
    [string]$Mode = "gate",
    [string]$DexHand = "inspire",
    [ValidateSet("RH", "LH")]
    [string]$Side = "RH",
    [string]$OakinkDataDir = "data/OakInk-v2",
    [string]$RunsRoot = "runs",
    [string]$AnalysisDir = "runs/analysis",
    [switch]$SkipTrain,
    [switch]$SkipSummary
)

if ($Mode -eq "smoke") {
    $Seeds = @(42)
    $NumEnvs = 128
    $MaxIterations = 120
    $EarlyStopEpochs = 120
} else {
    $Seeds = @(42, 142, 242)
    $NumEnvs = 512
    $MaxIterations = 1200
    $EarlyStopEpochs = 1200
}

$common = @(
    "task=ResDexHand",
    "rl_train=ResDexHandPPO",
    "side=$Side",
    "dexhand=$DexHand",
    "headless=true",
    "test=false",
    "num_envs=$NumEnvs",
    "max_iterations=$MaxIterations",
    "early_stop_epochs=$EarlyStopEpochs",
    "dataIndices=[oakink_auto_short]",
    "auto_oakink_short=True",
    "oakink_short_topk=1",
    "oakink_short_max_frames=180",
    "oakink_data_dir=$OakinkDataDir",
    "oakink_skip=2"
)

$ablations = @(
    @{ Name = "A0_pose_baseline"; Extra = @("task.env.usePointTarget=False", "task.env.usePtFlow=False", "task.env.useRegionGeom=False", "task.env.poseFallback=True") },
    @{ Name = "A1_ptpos"; Extra = @("task.env.usePointTarget=True", "task.env.usePtFlow=False", "task.env.useRegionGeom=False", "task.env.poseFallback=True") },
    @{ Name = "A2_ptpos_ptflow"; Extra = @("task.env.usePointTarget=True", "task.env.usePtFlow=True", "task.env.useRegionGeom=False", "task.env.poseFallback=True") },
    @{ Name = "A3_ptpos_ptflow_region_geom"; Extra = @("task.env.usePointTarget=True", "task.env.usePtFlow=True", "task.env.useRegionGeom=True", "task.env.poseFallback=True") }
)

foreach ($seed in $Seeds) {
    if (-not $SkipTrain) {
        foreach ($ab in $ablations) {
            $exp = "$($ab.Name)_s$seed"
            $args = @("main/rl/train.py") + $common + @("seed=$seed", "experiment=$exp") + $ab.Extra
            Write-Host "=== Running $exp ==="
            & python @args
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed at $exp (exit code: $LASTEXITCODE)"
            }
        }
    }
}

if (-not $SkipSummary) {
    $seedStr = ($Seeds -join ",")
    $summaryArgs = @(
        "main/rl/summarize_a0_a3_results.py",
        "--runs-root", $RunsRoot,
        "--analysis-dir", $AnalysisDir,
        "--seeds", $seedStr
    )
    Write-Host "=== Summarizing A0-A3 results to $AnalysisDir ==="
    & python @summaryArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Summary failed (exit code: $LASTEXITCODE)"
    }
}
