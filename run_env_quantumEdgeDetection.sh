#!/bin/bash -l

#SBATCH -p icelakegpu
#SBATCH -q normal
#SBATCH -J edgeDetection_test
#SBATCH -n 64
#SBATCH -N 1
#SBATCH -o edgeDetection_test-%j.o
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smerib00@estudiantes.unileon.es
#SBATCH --mem=0

# Activamos el entorno de conda
source /home/smerino/miniconda3/etc/profile.d/conda.sh
conda activate quantumEdgeDetection

python mainMask128.py

conda deactivate
