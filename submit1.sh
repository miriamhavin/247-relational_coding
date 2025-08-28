#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48GB
#SBATCH --nodes=1
##SBATCH --constraint=gpu80
##SBATCH --cpus-per-task=16
#SBATCH -o 'logs/%A.log'


if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate 247-main
elif [[ "$HOSTNAME" == *"della"* ]]
then
    echo "It's Della"
    module load anaconda3/2021.11
    source activate /home/kw1166/.conda/envs/247-main
else
    module load anacondapy
    source activate srm
fi

export TRANSFORMERS_OFFLINE=1

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python "$@"
fi
echo 'End time:' `date`
