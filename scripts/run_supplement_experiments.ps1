param(
    [ValidateSet("shared","dsac","all")]
    [string]$Mode = "all",
    [string]$Exe = ".\build\Release\CED_Schedule.exe",
    [string]$OutRoot = ".\results\supplement",
    [int[]]$Seeds = @(1,2,3,4,5,6,7,8,9,10)
)

$ErrorActionPreference = "Stop"

$scales = @(
    @{ Name = "T100"; DataFile = "data_matrix_100.txt"; Tnum = 100; Cnum = 100; Enum = 100; Dnum = 300 },
    @{ Name = "T200"; DataFile = "data_matrix_T200_E100_D300.txt"; Tnum = 200; Cnum = 100; Enum = 100; Dnum = 300 },
    @{ Name = "T500"; DataFile = "data_matrix_T500_E200_D800.txt"; Tnum = 500; Cnum = 200; Enum = 200; Dnum = 800 }
)

function Run-SharedBandit {
    foreach ($scale in $scales) {
        foreach ($seed in $Seeds) {
            $log = Join-Path $OutRoot ("shared\cchihh_shared_{0}_seed{1}.log" -f $scale.Name, $seed)
            $reward = Join-Path $OutRoot ("shared\reward_var_shared_{0}_seed{1}.csv" -f $scale.Name, $seed)
            $woff = Join-Path $OutRoot ("shared\weights_shared_off_{0}_seed{1}.csv" -f $scale.Name, $seed)
            $wseq = Join-Path $OutRoot ("shared\weights_shared_seq_{0}_seed{1}.csv" -f $scale.Name, $seed)
            $wdev = Join-Path $OutRoot ("shared\weights_shared_dev_{0}_seed{1}.csv" -f $scale.Name, $seed)
            New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($log)) | Out-Null
            & $Exe --solver CCHIHH --data_dir .\data --data_file $scale.DataFile `
                --tnum $scale.Tnum --mopt 5 --cnum $scale.Cnum --enum $scale.Enum --dnum $scale.Dnum `
                --generations 10000 --popsize 40 --nsubpop 8 --seed $seed `
                --stable --shared_bandit --log_every 50 `
                --reward_variance_log $reward `
                --cchihh_weight_log_offload $woff `
                --cchihh_weight_log_seq $wseq `
                --cchihh_weight_log_dev $wdev *> $log
        }
    }
}

function Run-Dsac {
    foreach ($scale in $scales) {
        foreach ($seed in $Seeds) {
            $log = Join-Path $OutRoot ("dsac\dsac_de_{0}_seed{1}.log" -f $scale.Name, $seed)
            New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($log)) | Out-Null
            & $Exe --solver DSAC-DE --data_dir .\data --data_file $scale.DataFile `
                --tnum $scale.Tnum --mopt 5 --cnum $scale.Cnum --enum $scale.Enum --dnum $scale.Dnum `
                --generations 10000 --popsize 40 --seed $seed --log_every 50 *> $log
        }
    }
}

switch ($Mode) {
    "shared" { Run-SharedBandit }
    "dsac" { Run-Dsac }
    "all" {
        Run-SharedBandit
        Run-Dsac
    }
}
