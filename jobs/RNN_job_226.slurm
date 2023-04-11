#!/bin/bash -l
#SBATCH -J RNN_226
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8100 &
python  ${path}/main.py 8101 &
python  ${path}/main.py 8102 &
python  ${path}/main.py 8103 &
python  ${path}/main.py 8104 &
python  ${path}/main.py 8105 &
python  ${path}/main.py 8106 &
python  ${path}/main.py 8107 &
python  ${path}/main.py 8108 &
python  ${path}/main.py 8109 &
python  ${path}/main.py 8110 &
python  ${path}/main.py 8111 &
python  ${path}/main.py 8112 &
python  ${path}/main.py 8113 &
python  ${path}/main.py 8114 &
python  ${path}/main.py 8115 &
python  ${path}/main.py 8116 &
python  ${path}/main.py 8117 &
python  ${path}/main.py 8118 &
python  ${path}/main.py 8119 &
python  ${path}/main.py 8120 &
python  ${path}/main.py 8121 &
python  ${path}/main.py 8122 &
python  ${path}/main.py 8123 &
python  ${path}/main.py 8124 &
python  ${path}/main.py 8125 &
python  ${path}/main.py 8126 &
python  ${path}/main.py 8127 &
python  ${path}/main.py 8128 &
python  ${path}/main.py 8129 &
python  ${path}/main.py 8130 &
python  ${path}/main.py 8131 &
python  ${path}/main.py 8132 &
python  ${path}/main.py 8133 &
python  ${path}/main.py 8134 &
python  ${path}/main.py 8135 &
wait
