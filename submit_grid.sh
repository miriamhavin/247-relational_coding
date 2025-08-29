#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -o 'logs/%A_%a.log'
# (Option A) You can add this header instead of passing --array on the CLI:
# #SBATCH --array=0-47%20

# -------------------- env --------------------
if [[ "$HOSTNAME" == *"tiger"* ]]; then
  module load anaconda && source activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]; then
  module load anaconda3/2021.11 && source activate /home/kw1166/.conda/envs/247-main
  # ^ If that env is not yours, point to your own conda env path.
else
  module load anacondapy && source activate srm
fi
export TRANSFORMERS_OFFLINE=1

# -------------------- grid --------------------
SIDS=(625 676 7170 798)
LAGS_LIST=('[-500,0,500]' '[-250,0,250]' '[-100,0,100]')
MINOCC_LIST=(10 20 40 50)

NS=${#SIDS[@]}
NL=${#LAGS_LIST[@]}
NM=${#MINOCC_LIST[@]}
TOT=$((NS*NL*NM))   # 48

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Run with: sbatch --array=0-$((TOT-1))%20 submit_grid.sh"
  exit 1
fi

i=${SLURM_ARRAY_TASK_ID}
if (( i < 0 || i >= TOT )); then
  echo "Array index $i out of range (0..$((TOT-1))). Exiting."
  exit 2
fi

# index decode: i -> (isid, ilag, imin)
isid=$(( i / (NL*NM) ))
rem=$(( i % (NL*NM) ))
ilag=$(( rem / NM ))
imin=$(( rem % NM ))

SID=${SIDS[$isid]}
LAGS=${LAGS_LIST[$ilag]}
MINOCC=${MINOCC_LIST[$imin]}

# tag: clean lags like [-500,0,500] -> -500_0_500
CLEAN_LAGS=$(echo "$LAGS" | tr -d '[] ' | sed 's/,/_/g')
TAG="sid${SID}_lags${CLEAN_LAGS}_min${MINOCC}"

echo "Requester: $USER"
echo "Node: $HOSTNAME"
echo "Start: $(date)"
echo "Grid index $i/$((TOT-1))  ->  SID=$SID  LAGS=$LAGS  MINOCC=$MINOCC"
echo "TAG: $TAG"

# -------------------- run --------------------
python scripts/tfsenc_main.py \
  --config-file configs/config.yml configs/${SID}-config.yml configs/gpt2-config.yml \
  --lags "$LAGS" \
  --min-occ $MINOCC \
  --output-dir-name "$TAG"

echo "End: $(date)"
