#!/bin/bash -l
#SBATCH -J RNN_254
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9108 &
python  ${path}/main.py 9109 &
python  ${path}/main.py 9110 &
python  ${path}/main.py 9111 &
python  ${path}/main.py 9112 &
python  ${path}/main.py 9113 &
python  ${path}/main.py 9114 &
python  ${path}/main.py 9115 &
python  ${path}/main.py 9116 &
python  ${path}/main.py 9117 &
python  ${path}/main.py 9118 &
python  ${path}/main.py 9119 &
python  ${path}/main.py 9120 &
python  ${path}/main.py 9121 &
python  ${path}/main.py 9122 &
python  ${path}/main.py 9123 &
python  ${path}/main.py 9124 &
python  ${path}/main.py 9125 &
python  ${path}/main.py 9126 &
python  ${path}/main.py 9127 &
python  ${path}/main.py 9128 &
python  ${path}/main.py 9129 &
python  ${path}/main.py 9130 &
python  ${path}/main.py 9131 &
python  ${path}/main.py 9132 &
python  ${path}/main.py 9133 &
python  ${path}/main.py 9134 &
python  ${path}/main.py 9135 &
python  ${path}/main.py 9136 &
python  ${path}/main.py 9137 &
python  ${path}/main.py 9138 &
python  ${path}/main.py 9139 &
python  ${path}/main.py 9140 &
python  ${path}/main.py 9141 &
python  ${path}/main.py 9142 &
python  ${path}/main.py 9143 &
wait
