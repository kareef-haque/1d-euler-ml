#!/bin/bash
# Large-CFL pairwise models (split transport vs transport) on one GPU on Rivanna. Submit from the repo root:
#     sbatch slurm/train_split_gpu.slurm                                         # split_c4_long + trans_c4_long, seeds 0 1 2
#     sbatch --export=ALL,VARIANTS=split_c4_long slurm/train_split_gpu.slurm     # one variant
#     sbatch --export=ALL,VARIANTS="split_c4 split_c4_nolim" slurm/train_split_gpu.slurm   # any list, in that order
#     sbatch --export=ALL,SEEDS=0 slurm/train_split_gpu.slurm                    # one seed
#     sbatch --export=ALL,OUT=ablation_split2 slurm/train_split_gpu.slurm        # another output folder
#     sbatch --export=ALL,QUICK=1 slurm/train_split_gpu.slurm                    # tiny schedule per variant (minutes)
# Variant names are the keys of VARIANTS in ablate_nfv.py. Each (variant, seed) runs in its own python process, so
# its GPU memory is freed when it exits. A run that crashes is logged and the next one still runs; the job exits 1
# if any crashed. Runs already in OUT (a result.json exists) are skipped, so a resubmit continues where it stopped.
# A run that hits NaN is recorded as failed in its result.json (ablate_nfv uses max_restarts=0), not as a crash.
# Before the first run, on a login node:
#     sinfo -p gpu -o "%N %G %t"          # GPU types and which nodes are idle/mix
#     ls $CONTAINERDIR | grep -i torch    # container versions; put one in SIF below
# Datasets are generated on the first run into DATA (exact Riemann solutions, CPU) and reused after that.
# Output: OUT/<variant>_s<seed>/ (log.jsonl, checkpoints, result.json), OUT/summary.json, ncfl-<job>.out
#SBATCH -A sds_baek_energetic           # <- your allocation (or sbatch -A <alloc>, which overrides this line)
#SBATCH -p gpu
#SBATCH --gres=gpu:1                     # any GPU; gpu:a100:1 for an A100 (names from sinfo's %G)
#SBATCH -c 4
#SBATCH --mem=32G                        # datasets are a few hundred MB at most
#SBATCH -t 0-12:00:00                    # ⚑ not measured on GPU; 3-stage runs took 0.3-0.8 h each on a desktop CPU,
#                                        #   the LONG schedule adds a stage; 6 runs + margin. Shorten once timed.
#SBATCH -J ncfl
#SBATCH -o ncfl-%j.out
#SBATCH -e ncfl-%j.err

VARIANTS=${VARIANTS:-"split_c4_long trans_c4_long"}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-ablation_split}
DATA=${DATA:-generated_nfv}
[ "${QUICK:-0}" = "1" ] && OUT=${OUT}_quick
cd "$SLURM_SUBMIT_DIR"
# fail fast on the usual setup mistakes, before any training starts
[ -f ablate_nfv.py ] && [ -f train_nfv.py ]      || { echo "ERROR: submit from the repo root (no ablate_nfv.py in $PWD)"; exit 1; }
for V in $VARIANTS; do
    grep -q "^    '$V':" ablate_nfv.py              || { echo "ERROR: unknown variant $V (see VARIANTS in ablate_nfv.py)"; exit 1; }
done

module purge
module load apptainer pytorch
SIF=${SIF:-$CONTAINERDIR/pytorch-2.13.0.sif}     # <- version used on Rivanna 2026-10-31 (or sbatch --export=ALL,SIF=...)
[ -f "$SIF" ] || { echo "ERROR: container $SIF not found. Available:"; ls "$CONTAINERDIR" | grep -i torch; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "variants: $VARIANTS   seeds: $SEEDS   out: $OUT   data: $DATA   quick: ${QUICK:-0}   container: $SIF   node: $(hostname)"

# numpy, scipy (exact Riemann solver) and matplotlib; installed to ~/.local once if the container lacks them
apptainer exec --nv $SIF python -c "import numpy, scipy, matplotlib" 2>/dev/null \
    || apptainer exec --nv $SIF pip install --user numpy scipy matplotlib
apptainer exec --nv $SIF python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" \
    || { echo "ERROR: torch cannot see the GPU"; exit 1; }

if [ "${QUICK:-0}" = "1" ]; then
    EXTRA="--force_schedule 20:1e-4:10:10,20:1e-5:50:50 --n_train 32 --batch_size 16 --n_eval 8 --eval_every 10"
    DATA=${DATA}_quick
else
    EXTRA=""
fi

FAILED=""
for V in $VARIANTS; do
    for S in $SEEDS; do
        echo "================ $V seed $S: start $(date) ================"
        apptainer exec --nv $SIF python -u ablate_nfv.py --variants $V --seeds $S --out $OUT --data_dir $DATA \
            --device cuda $EXTRA
        rc=$?
        echo "================ $V seed $S: exit $rc $(date) ================"
        [ $rc -eq 0 ] || FAILED="$FAILED $V/s$S"
    done
done

# one summary over everything in this job (finished runs are read back from their result.json, not retrained)
echo "================ summary ================"
apptainer exec --nv $SIF python -u ablate_nfv.py --variants $(echo $VARIANTS | tr ' ' ',') \
    --seeds $(echo $SEEDS | tr ' ' ',') --out $OUT --data_dir $DATA --device cpu $EXTRA
apptainer exec --nv $SIF python -u plot_sod_ablation.py "$OUT/*/model_final.pt" --out $OUT/sod.png
apptainer exec --nv $SIF python -u plot_sod_ablation.py "$OUT/*/model_final.pt" --out $OUT/sod_prim.png \
    --vars prim --xlim 0.15 0.3
[ -z "$FAILED" ] || { echo "CRASHED runs:$FAILED"; exit 1; }
