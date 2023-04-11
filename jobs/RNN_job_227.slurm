#!/bin/bash -l
#SBATCH -J RNN_227
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8136 &
python  ${path}/main.py 8137 &
python  ${path}/main.py 8138 &
python  ${path}/main.py 8139 &
python  ${path}/main.py 8140 &
python  ${path}/main.py 8141 &
python  ${path}/main.py 8142 &
python  ${path}/main.py 8143 &
python  ${path}/main.py 8144 &
python  ${path}/main.py 8145 &
python  ${path}/main.py 8146 &
python  ${path}/main.py 8147 &
python  ${path}/main.py 8148 &
python  ${path}/main.py 8149 &
python  ${path}/main.py 8150 &
python  ${path}/main.py 8151 &
python  ${path}/main.py 8152 &
python  ${path}/main.py 8153 &
python  ${path}/main.py 8154 &
python  ${path}/main.py 8155 &
python  ${path}/main.py 8156 &
python  ${path}/main.py 8157 &
python  ${path}/main.py 8158 &
python  ${path}/main.py 8159 &
python  ${path}/main.py 8160 &
python  ${path}/main.py 8161 &
python  ${path}/main.py 8162 &
python  ${path}/main.py 8163 &
python  ${path}/main.py 8164 &
python  ${path}/main.py 8165 &
python  ${path}/main.py 8166 &
python  ${path}/main.py 8167 &
python  ${path}/main.py 8168 &
python  ${path}/main.py 8169 &
python  ${path}/main.py 8170 &
python  ${path}/main.py 8171 &
wait
