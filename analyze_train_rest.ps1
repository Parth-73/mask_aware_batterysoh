$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$train = Import-Csv (Join-Path $root 'pre processed ready to use data/train.csv')
$source = Import-Csv (Join-Path $root 'benchmark-repository/CALCE-CX2/outputs/cx2_features_all.csv')
$lookup = @{}
foreach ($r in $source) { $lookup[$r.cycle_id] = $r }
$out = Join-Path $root 'train_rest_analysis'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$transitions = foreach ($group in ($train | Group-Object battery)) {
    $rows = @($group.Group | Sort-Object { [int]$_.cycle_ordinal })
    for ($i = 1; $i -lt $rows.Count; $i++) {
        $p = $rows[$i-1]; $r = $rows[$i]
        $gap = [int]$r.cycle_ordinal - [int]$p.cycle_ordinal
        $s = $lookup[$p.cycle_id]
        if (!$s) { throw "Missing source cycle $($p.cycle_id)" }
        $hours = ([datetime]$r.cycle_start - [datetime]$s.end).TotalHours
        $delta = 100 * ([double]$r.soh - [double]$p.soh)
        $nextDelta = $null
        if ($i+1 -lt $rows.Count -and [int]$rows[$i+1].cycle_ordinal -eq [int]$r.cycle_ordinal + 1) {
            $nextDelta = 100 * ([double]$rows[$i+1].soh - [double]$r.soh)
        }
        $status = 'ordinary_transition'
        if ($gap -ne 1) { $status = 'missing_cycles_not_rest_verified' }
        elseif ($hours -lt 0) { $status = 'timestamp_overlap' }
        elseif ($hours -ge 6 -and $delta -ge 0.2) { $status = 'increase_after_long_gap' }
        elseif ($hours -ge 6 -and $delta -le -0.2) { $status = 'decrease_after_long_gap' }
        elseif ($hours -ge 6) { $status = 'long_gap_small_change' }
        [pscustomobject]@{
            battery=$r.battery; previous_cycle=[int]$p.cycle_ordinal; cycle=[int]$r.cycle_ordinal
            gap_start=$s.end; gap_end=$r.cycle_start; unobserved_gap_h=$hours
            cycle_gap=$gap; previous_soh_percent=100*[double]$p.soh; soh_percent=100*[double]$r.soh
            change_pp=$delta; next_cycle_change_pp=$nextDelta
            reversal_next_cycle=($null -ne $nextDelta -and [math]::Abs($delta) -ge 0.2 -and [math]::Abs($nextDelta) -ge 0.2 -and $delta*$nextDelta -lt 0)
            source_changed=($p.source_identifiers -ne $r.source_identifiers)
            classification=$status
        }
    }
}
$transitions | Export-Csv (Join-Path $out 'all_transitions.csv') -NoTypeInformation
$events = @($transitions | Where-Object { $_.classification -in @('increase_after_long_gap','decrease_after_long_gap') } | Sort-Object { [math]::Abs($_.change_pp) } -Descending)
$events | Export-Csv (Join-Path $out 'rest_associated_events.csv') -NoTypeInformation
$transitions | Where-Object { $_.cycle_gap -ne 1 -and [math]::Abs($_.change_pp) -ge 0.2 } | Export-Csv (Join-Path $out 'missing_cycle_events.csv') -NoTypeInformation
@'
Input: pre processed ready to use data/train.csv (training rows only).
Source end times: benchmark-repository/CALCE-CX2/outputs/cx2_features_all.csv.
Method: sort within battery by original cycle ordinal. Compare SOH of consecutive retained rows.
Candidate long rest: at least 6 hours between prior cycle END and current cycle START, with consecutive original cycle ordinals.
Material change: absolute SOH change at least 0.2 percentage points (an exploratory threshold, not a calibrated noise limit).
Fluctuation flag: next consecutive training cycle reverses direction by at least 0.2 percentage points.
Missing original cycles are exported separately: elapsed time across them cannot be interpreted as rest.
Gaps are unobserved intervals, not verified zero-current rest. Source changes, protocol, temperature and measurement effects can explain changes. Causation is not established.
SOH is discharge capacity / 1.35 Ah. All changes are percentage points, not relative percent.
'@ | Set-Content (Join-Path $out 'README.txt')
$events | Select-Object battery,previous_cycle,cycle,gap_start,gap_end,@{n='gap_h';e={[math]::Round($_.unobserved_gap_h,2)}},@{n='change_pp';e={[math]::Round($_.change_pp,3)}},reversal_next_cycle | Format-Table -AutoSize
$transitions | Group-Object classification | Select-Object Name,Count | Format-Table -AutoSize
