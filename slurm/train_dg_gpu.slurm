#!/bin/bash
# Learned-flux DG (euler1d/dg) on one GPU on Rivanna: train, then evaluate. Submit from the repo root:
#     sbatch slurm/train_dg_gpu.slurm                                          # p1_scalar p1_char, seeds 0 1 2
#     sbatch --export=ALL,ARMS="p2_scalar p2_char" slurm/train_dg_gpu.slurm    # any list of p<degree>_<mode>, in that order
#     sbatch --export=ALL,SEEDS=0 slurm/train_dg_gpu.slurm                     # one seed
#     sbatch --export=ALL,S_MIN=1.0 slurm/train_dg_gpu.slurm                   # dissipation floor (1 = Rusanov, positivity-safe)
#     sbatch --export=ALL,OUT=runs_dg2 slurm/train_dg_gpu.slurm                # another output folder
#     sbatch --export=ALL,QUICK=1 slurm/train_dg_gpu.slurm                     # tiny schedule per arm (minutes)
# Arms: p1_scalar, p1_char, p2_scalar, p2_char (mode: scalar = one dissipation scale, char = one per wave).
# Each (arm, seed) trains in its own python process (GPU memory freed on exit), then eval_dg.py runs on its
# model_final.pt (test split, Sod / Lax / 123, smooth convergence order, Sod plot). A run that crashes is logged and
# the next one still runs; the job exits 1 if any crashed. A run whose eval.json exists is skipped, so a resubmit
# continues where it stopped. train_dg.py restarts an attempt on a non-finite loss (--max_restarts 3).
# Before the first run, on a login node:
#     sinfo -p gpu -o "%N %G %t"          # GPU types and which nodes are idle/mix
#     ls $CONTAINERDIR | grep -i torch    # container versions; put one in SIF below
# Datasets are generated on the first run into DATA (exact Riemann projections, CPU) and reused after that.
# The DG code is float64: fast on A100-class GPUs, slow on consumer GPUs (ask for gpu:a100:1 if available).
# Output: OUT/<arm>_s<seed>/ (log.jsonl, model_*.pt, eval.json, sod.png), ncdg-<job>.out
#SBATCH -A sds_baek_energetic           # <- your allocation (or sbatch -A <alloc>, which overrides this line)
#SBATCH -p gpu
#SBATCH --gres=gpu:1                     # any GPU; gpu:a100:1 for an A100 (names from sinfo's %G)
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00                    # ⚑ not measured on GPU (CPU smoke: ~0.7 s per training step at nx 40, nt 20);
#                                        #   6 runs + margin. Shorten once a real run is timed.
#SBATCH -J ncdg
#SBATCH -o ncdg-%j.out
#SBATCH -e ncdg-%j.err

ARMS=${ARMS:-"p1_scalar p1_char"}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-runs_dg}
DATA=${DATA:-generated_dg}
S_MIN=${S_MIN:-0.0}
cd "$SLURM_SUBMIT_DIR"
[ -f train_dg.py ] && [ -f eval_dg.py ]          || { echo "ERROR: submit from the repo root (no train_dg.py in $PWD)"; exit 1; }
for A in $ARMS; do
    [[ "$A" =~ ^p[0-9]+_(scalar|char)$ ]]       || { echo "ERROR: arm $A is not p<degree>_<scalar|char>"; exit 1; }
done

module purge
module load apptainer pytorch
SIF=${SIF:-$CONTAINERDIR/pytorch-2.13.0.sif}     # <- version used on Rivanna 2026-10-31 (or sbatch --export=ALL,SIF=...)
[ -f "$SIF" ] || { echo "ERROR: container $SIF not found. Available:"; ls "$CONTAINERDIR" | grep -i torch; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "arms: $ARMS   seeds: $SEEDS   s_min: $S_MIN   out: $OUT   data: $DATA   quick: ${QUICK:-0}   container: $SIF   node: $(hostname)"

# numpy, scipy (exact Riemann solver) and matplotlib; installed to ~/.local once if the container lacks them
apptainer exec --nv $SIF python -c "import numpy, scipy, matplotlib" 2>/dev/null \
    || apptainer exec --nv $SIF pip install --user numpy scipy matplotlib
apptainer exec --nv $SIF python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))" \
    || { echo "ERROR: torch cannot see the GPU"; exit 1; }

if [ "${QUICK:-0}" = "1" ]; then
    TRAIN_EXTRA="--schedule 20:1e-3:20:10,10:5e-4:40:20 --n_train 32 --batch 8 --n_eval 8 --eval_nx 60 --eval_every 10"
    EVAL_EXTRA="--n_test 8 --test_nx 60 --named_N 60"
    OUT=${OUT}_quick
    DATA=${DATA}_quick
else
    TRAIN_EXTRA=""
    EVAL_EXTRA=""
fi

FAILED=""
for A in $ARMS; do
    P=${A%%_*}; P=${P#p}; MODE=${A#*_}
    for S in $SEEDS; do
        RUN=$OUT/${A}_s$S
        if [ -f "$RUN/eval.json" ]; then echo "skip $RUN (eval.json exists)"; continue; fi
        echo "================ $A seed $S: start $(date) ================"
        apptainer exec --nv $SIF python -u train_dg.py --p $P --mode $MODE --seed $S --s_min $S_MIN \
            --data_dir $DATA --out $RUN --device cuda $TRAIN_EXTRA \
        && apptainer exec --nv $SIF python -u eval_dg.py --model $RUN/model_final.pt --data_dir $DATA \
            --plot $RUN/sod.png $EVAL_EXTRA
        rc=$?
        echo "================ $A seed $S: exit $rc $(date) ================"
        [ $rc -eq 0 ] || FAILED="$FAILED $A/s$S"
    done
done

# one table over everything in OUT (reads the eval.json files)
echo "================ summary ================"
apptainer exec --nv $SIF python - "$OUT" <<'EOF'
import glob, json, os, sys
rows = sorted(glob.glob(os.path.join(sys.argv[1], '*', 'eval.json')))
print(f"{'run':18s} {'learned':>10s} {'DG-HLLC':>10s} {'x vs HLLC':>9s} {'wins':>5s} {'bad':>4s} | {'sod':>9s} {'lax':>9s} {'123':>9s} | rate")
for f in rows:
    r = json.load(open(f))
    t, n, c = r['test'], r['named'], r['convergence']
    print(f"{os.path.basename(os.path.dirname(f)):18s} {t['learned']['err']:10.3e} {t['DG-HLLC']['err']:10.3e} "
          f"{t['DG-HLLC']['err'] / t['learned']['err']:9.2f} {100 * t['learned']['win_vs_hllc']:4.0f}% {t['learned']['bad']:4d} | "
          f"{n['sod']['learned']:9.2e} {n['lax']['learned']:9.2e} {n['123']['learned']:9.2e} | {c['learned']['rate']:.2f}")
EOF
[ -z "$FAILED" ] || { echo "CRASHED runs:$FAILED"; exit 1; }
