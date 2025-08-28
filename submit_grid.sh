#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -o 'logs/%A_%a.log'

# -------------------- env --------------------
if [[ "$HOSTNAME" == *"tiger"* ]]; then
  module load anaconda && source activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]; then
  module load anaconda3/2021.11 && source activate /home/kw1166/.conda/envs/247-main
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
TOT=$((NS*NL*NM))

# Bounds check (useful when testing interactively)
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Run with: sbatch --array=0-$((TOT-1)) submit_grid.sh scripts/tfsenc_main.py"
  exit 1
fi
i=${SLURM_ARRAY_TASK_ID}

# index decode: i -> (isid, ilag, imin)
isid=$(( i / (NL*NM) ))
rem=$(( i % (NL*NM) ))
ilag=$(( rem / NM ))
imin=$(( rem % NM ))

SID=${SIDS[$isid]}
LAGS=${LAGS_LIST[$ilag]}
MINOCC=${MINOCC_LIST[$imin]}

# nice tag to keep runs separated on disk
TAG="sid${SID}_lags$(echo $LAGS | tr -dc 0-9n-)_min${MINOCC}"

echo "Requester: $USER"
echo "Node: $HOSTNAME"
echo "Start: $(date)"
echo "Grid index $i / $((TOT-1))  ->  SID=$SID  LAGS=$LAGS  MINOCC=$MINOCC"
echo "TAG: $TAG"

# -------------------- run --------------------
# NOTE: we pass three config files like you do now, plus CLI overrides.
python scripts/tfsenc_main.py \
  --config-file configs/config.yml configs/${SID}-config.yml configs/gpt2-config.yml \
  --lags "$LAGS" \
  --min-occ $MINOCC \
  --output_dir_name "$TAG"

echo "End: $(date)"
