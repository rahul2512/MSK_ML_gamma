#!/bin/bash -l
#SBATCH -J CNN_6
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 120 &
python  ${path}/main.py 121 &
python  ${path}/main.py 122 &
python  ${path}/main.py 123 &
python  ${path}/main.py 124 &
python  ${path}/main.py 125 &
python  ${path}/main.py 126 &
python  ${path}/main.py 127 &
python  ${path}/main.py 128 &
python  ${path}/main.py 129 &
python  ${path}/main.py 130 &
python  ${path}/main.py 131 &
python  ${path}/main.py 132 &
python  ${path}/main.py 133 &
python  ${path}/main.py 134 &
python  ${path}/main.py 135 &
python  ${path}/main.py 136 &
python  ${path}/main.py 137 &
python  ${path}/main.py 138 &
python  ${path}/main.py 139 &
python  ${path}/main.py 140 &
python  ${path}/main.py 141 &
python  ${path}/main.py 142 &
python  ${path}/main.py 143 &
wait
