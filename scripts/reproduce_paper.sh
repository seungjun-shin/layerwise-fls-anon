#!/usr/bin/env bash
# One-to-one command entry points for every experimental table and figure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cuda:0}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

TARGET="${1:-list}"

print_command() {
    printf '  +'
    printf ' %q' "$@"
    printf '\n'
}

run_command() {
    print_command "$@"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

run_if_missing() {
    local marker="$1"
    shift
    if [[ -e "$marker" ]]; then
        printf '  = skip completed artifact: %s\n' "$marker"
        return
    fi
    run_command "$@"
}

run_python() {
    run_command "$PYTHON_BIN" "$@"
}

run_jobs() {
    local jobs="$1"
    run_python scripts/gpu_pool_run.py --jobs "$jobs" --gpus "$GPU_LIST" --python "$PYTHON_BIN"
}

list_targets() {
    cat <<'EOF'
Main manuscript tables:
  main-table-1 .. main-table-6
Main manuscript experimental figures:
  main-figure-2 .. main-figure-12
Supplementary tables:
  supp-table-I .. supp-table-XXV
Convenience:
  all    run each unique experiment family and generate every figure

Examples:
  scripts/reproduce_paper.sh --dry-run main-table-1
  GPU_LIST=0,1 ANALYSIS_DEVICE=cuda:0 scripts/reproduce_paper.sh supp-table-XIII
EOF
}

run_grid_calibration() {
    run_jobs configs/jobs_val_regrid_seed0.txt
    run_jobs configs/jobs_val_regrid_seed12.txt
    run_python scripts/build_val_grid_summary.py
}

run_validation_protocol() {
    run_jobs configs/jobs_val_protocol.txt
    run_python scripts/summarize_val_protocol.py
}

run_confound_family() {
    run_jobs configs/jobs_val_confound.txt
    run_jobs configs/jobs_val_cf_ext.txt
    run_python scripts/build_val_confound_summary.py
}

generate_core_diagnostic_jobs() {
    local jobs="outputs/_paper_reproduction_jobs/core_diagnostics.txt"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '  + generate %s (six validation-controlled arms, seeds 0--4)\n' "$jobs"
    else
        mkdir -p "$(dirname "$jobs")"
        : > "$jobs"
        for seed in 0 1 2 3 4; do
            printf '%s\n' "configs/experiment_49_global_matched_seed_expansion.yaml training.seed=$seed training.max_epochs=80 data.val_fraction=0.1 data.val_seed=0 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_global_c0p25" >> "$jobs"
            printf '%s\n' "configs/imp_pos_base.yaml training.seed=$seed training.max_epochs=80 fls.position=8 fls.output_multiplier=1.0 fls.lr_compensation_multiplier=1.0 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_identity" >> "$jobs"
            printf '%s\n' "configs/imp_pos_base.yaml training.seed=$seed training.max_epochs=80 fls.position=2 fls.output_multiplier=0.0625 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_boundary_cut2" >> "$jobs"
            printf '%s\n' "configs/imp_pos_base.yaml training.seed=$seed training.max_epochs=80 fls.position=8 fls.output_multiplier=0.0625 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_boundary_cut8" >> "$jobs"
            printf '%s\n' "configs/imp_pos_base.yaml training.seed=$seed training.max_epochs=80 fls.position=8 fls.output_multiplier=1.0 fls.lr_compensation_multiplier=0.0625 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_lronly_cut8" >> "$jobs"
            printf '%s\n' "configs/val_cf_nocomp.yaml training.seed=$seed training.max_epochs=80 training.save_checkpoints=true training.checkpoint_at_accs=[] experiment.name=val60_core_nocomp_cut8" >> "$jobs"
        done
    fi
    run_jobs "$jobs"
}

run_core_diagnostics() {
    generate_core_diagnostic_jobs
    run_python scripts/run_boundary_protocol_diagnostics.py --preset main_pair --split train --max-samples 512 --sample-seed 2001 --device "$ANALYSIS_DEVICE"
    run_python scripts/run_boundary_protocol_diagnostics.py --preset confound4way --split train --max-samples 512 --sample-seed 2001 --device "$ANALYSIS_DEVICE" --output-suffix confound4way
    run_python scripts/compute_update_pressure_diagnostics.py --preset confound4way --checkpoint checkpoint_best.pt --split train --max-samples 256 --sample-seed 2001 --device "$ANALYSIS_DEVICE" --output-dir outputs/_val_preserve_all_20260801/diagnostics/update_pressure
    run_python scripts/compute_discriminating_diagnostic.py --protocol val60 --split test --max-samples 5000 --device "$ANALYSIS_DEVICE" --out outputs/_val_preserve_all_20260801/diagnostics/discriminating_diagnostic.md
}

run_depthwise_diagnostics() {
    generate_core_diagnostic_jobs
    run_python scripts/run_depth_profiles.py --preset validation_core --device "$ANALYSIS_DEVICE" --max-samples 256
}

run_robustness() {
    generate_core_diagnostic_jobs
    run_python scripts/eval_robustness.py --experiment val60_core_global_c0p25 --experiment val60_core_boundary_cut8 --seeds 0 1 2 --device "$ANALYSIS_DEVICE" --batch-size 256 --data-root data --no-download
}

run_iso_accuracy() {
    run_jobs configs/jobs_iso_acc_cut8.txt
    run_jobs configs/jobs_iso_vgg_cut15.txt
    run_python scripts/compare_iso_acc_repr_val.py
    run_python scripts/compare_iso_acc_repr_vgg_val.py
}

run_vgg_validation() {
    run_jobs configs/jobs_val_vgg_stabilization.txt
    run_jobs configs/jobs_val_vgg_depth_paired.txt
    run_jobs configs/jobs_val_vgg_lr_fine.txt
    run_jobs configs/jobs_val_vgg_seed_replication.txt
    run_python scripts/summarize_val_vgg_stabilization.py
    run_python scripts/summarize_val_vgg_depth_paired.py
    run_python scripts/summarize_val_vgg_lr_fine.py
    run_python scripts/summarize_val_vgg_seed_replication.py
}

run_tiny_validation() {
    local phase1="outputs/_paper_reproduction_jobs/tiny_phase1.txt"
    local phase2="outputs/_paper_reproduction_jobs/tiny_phase2.txt"
    local summary="outputs/_summaries/tiny72_20260731"
    run_python scripts/build_val_tiny72_jobs.py --output "$phase1"
    run_jobs "$phase1"
    run_python scripts/summarize_val_tiny72.py --stage phase1 --outputs-root outputs --output-dir "$summary" --phase2-jobs "$phase2"
    run_jobs "$phase2"
    run_python scripts/summarize_val_tiny72.py --stage final --outputs-root outputs --output-dir "$summary"
}

run_generic_family() {
    local family="$1"
    run_python paper_reproduction/scripts/generic_queue.py \
        --manifest "paper_reproduction/manifests/${family}_jobs.json" \
        --state "paper_reproduction/manifests/${family}_queue_state.json" \
        --run-manifest "paper_reproduction/manifests/${family}_run_manifest.csv" \
        --failed-manifest "paper_reproduction/manifests/${family}_failed_runs.csv" \
        --gpus "$GPU_LIST" --python "$PYTHON_BIN"
}

run_frozen_families() {
    run_python paper_reproduction/scripts/build_jobs.py
    run_python paper_reproduction/scripts/build_transfer_jobs.py
    run_python paper_reproduction/scripts/build_head_norm_jobs.py
    run_python paper_reproduction/scripts/build_vgg_depth_extension.py
    run_python paper_reproduction/scripts/build_scope_preservation.py

    run_python paper_reproduction/scripts/local_queue.py --manifest paper_reproduction/manifests/practical_main_jobs.json --only-block reconstruction --gpus "$GPU_LIST" --python "$PYTHON_BIN"
    run_python paper_reproduction/scripts/local_queue.py --manifest paper_reproduction/manifests/practical_main_jobs.json --only-block confirmatory --gpus "$GPU_LIST" --python "$PYTHON_BIN"
    run_generic_family practical_transfer
    run_generic_family head_norm
    run_generic_family vgg_depth_extension
    run_generic_family scope_preservation

    local family
    for family in practical_main_reconstruction practical_main_confirmatory practical_transfer head_norm vgg_depth_extension scope_preservation; do
        run_python paper_reproduction/scripts/evaluate_families_once.py --family "$family" --device "$ANALYSIS_DEVICE"
    done
    run_if_missing paper_reproduction/analysis/mechanism/mechanism_bundle_complete.json "$PYTHON_BIN" paper_reproduction/scripts/mechanism_analysis.py --device "$ANALYSIS_DEVICE"
    run_if_missing paper_reproduction/analysis/statistics/accuracy_bundle_complete.json "$PYTHON_BIN" paper_reproduction/scripts/assemble_accuracy_results.py
    run_python paper_reproduction/scripts/statistics.py
    run_python paper_reproduction/scripts/make_protocol_artifacts.py
}

run_controlled_ten_seed() {
    run_validation_protocol
    run_jobs configs/jobs_val_vgg_stabilization.txt
    run_jobs configs/jobs_val_vgg_seed_replication.txt
    run_jobs configs/_jobs/table1_controlled_n10_extension.txt
    run_python paper_reproduction/scripts/analyze_table1_controlled_n10.py --raw-root paper_reproduction/raw_metrics
}

run_budget_schedule() {
    run_command env B_SCOPE=budget_schedule B_GPUS="$GPU_LIST" bash scripts/launch_B_budget_ladder.sh
    run_python scripts/summarize_budget_schedule.py --outputs-root outputs --output-dir results/paper
}

run_focused_grid() {
    run_grid_calibration
}

run_vgg_lr_grid() {
    run_vgg_validation
}

run_target() {
    case "$1" in
        main-table-1) run_controlled_ten_seed; run_frozen_families ;;
        main-table-2) run_validation_protocol; run_confound_family ;;
        main-table-3) run_frozen_families ;;
        main-table-4) run_frozen_families ;;
        main-table-5) run_core_diagnostics; run_iso_accuracy ;;
        main-table-6) run_confound_family; run_core_diagnostics ;;

        main-figure-2) run_grid_calibration; run_python scripts/plot_global_heatmap_val.py ;;
        main-figure-3) run_grid_calibration; run_python scripts/plot_location_sensitivity_val.py ;;
        main-figure-4) run_validation_protocol; run_python scripts/plot_depth_profile.py ;;
        main-figure-5) run_validation_protocol; run_vgg_validation; run_tiny_validation; run_python scripts/plot_depth_profile_multiarch.py ;;
        main-figure-6) run_validation_protocol; run_python scripts/plot_validation_seed_comparison.py ;;
        main-figure-7) run_iso_accuracy; run_python scripts/plot_iso_acc_repr.py ;;
        main-figure-8) run_depthwise_diagnostics; run_python scripts/plot_depth_repr_profile.py ;;
        main-figure-9) run_core_diagnostics; run_python scripts/plot_update_pressure.py ;;
        main-figure-10) run_grid_calibration; run_confound_family; run_python scripts/plot_c1_val.py ;;
        main-figure-11) run_confound_family; run_python scripts/plot_factorial_val.py ;;
        main-figure-12) run_frozen_families; run_python scripts/plot_scope_stress.py ;;

        supp-table-I) run_validation_protocol ;;
        supp-table-II) run_validation_protocol ;;
        supp-table-III) run_core_diagnostics ;;
        supp-table-IV) run_core_diagnostics ;;
        supp-table-V) run_confound_family ;;
        supp-table-VI) run_confound_family ;;
        supp-table-VII) run_robustness ;;
        supp-table-VIII) run_vgg_validation ;;
        supp-table-IX) run_vgg_validation ;;
        supp-table-X) run_tiny_validation ;;
        supp-table-XI) run_tiny_validation ;;
        supp-table-XII) run_depthwise_diagnostics ;;
        supp-table-XIII) run_controlled_ten_seed ;;
        supp-table-XIV) run_frozen_families ;;
        supp-table-XV) run_frozen_families ;;
        supp-table-XVI) run_frozen_families ;;
        supp-table-XVII) run_frozen_families ;;
        supp-table-XVIII) run_frozen_families ;;
        supp-table-XIX) run_frozen_families ;;
        supp-table-XX) run_frozen_families ;;
        supp-table-XXI) run_frozen_families ;;
        supp-table-XXII) run_budget_schedule ;;
        supp-table-XXIII) run_focused_grid ;;
        supp-table-XXIV) run_vgg_lr_grid ;;
        supp-table-XXV) run_iso_accuracy ;;
        *) printf 'Unknown target: %s\n' "$1" >&2; list_targets >&2; exit 2 ;;
    esac
}

run_all() {
    run_grid_calibration
    run_validation_protocol
    run_confound_family
    run_core_diagnostics
    run_depthwise_diagnostics
    run_robustness
    run_iso_accuracy
    run_vgg_validation
    run_tiny_validation
    run_controlled_ten_seed
    run_frozen_families
    run_budget_schedule
    run_python scripts/plot_global_heatmap_val.py
    run_python scripts/plot_location_sensitivity_val.py
    run_python scripts/plot_depth_profile.py
    run_python scripts/plot_depth_profile_multiarch.py
    run_python scripts/plot_validation_seed_comparison.py
    run_python scripts/plot_iso_acc_repr.py
    run_python scripts/plot_depth_repr_profile.py
    run_python scripts/plot_update_pressure.py
    run_python scripts/plot_c1_val.py
    run_python scripts/plot_factorial_val.py
    run_python scripts/plot_scope_stress.py
}

case "$TARGET" in
    list|--list|-h|--help) list_targets ;;
    all) run_all ;;
    *) run_target "$TARGET" ;;
esac
