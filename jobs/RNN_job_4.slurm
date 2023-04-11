#!/bin/bash -l
#SBATCH -J RNN_4
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 108 &
python  ${path}/main.py 109 &
python  ${path}/main.py 110 &
python  ${path}/main.py 111 &
python  ${path}/main.py 112 &
python  ${path}/main.py 113 &
python  ${path}/main.py 114 &
python  ${path}/main.py 115 &
python  ${path}/main.py 116 &
python  ${path}/main.py 117 &
python  ${path}/main.py 118 &
python  ${path}/main.py 119 &
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
