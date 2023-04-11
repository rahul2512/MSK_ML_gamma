#!/bin/bash -l
#SBATCH -J RNN_255
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9144 &
python  ${path}/main.py 9145 &
python  ${path}/main.py 9146 &
python  ${path}/main.py 9147 &
python  ${path}/main.py 9148 &
python  ${path}/main.py 9149 &
python  ${path}/main.py 9150 &
python  ${path}/main.py 9151 &
python  ${path}/main.py 9152 &
python  ${path}/main.py 9153 &
python  ${path}/main.py 9154 &
python  ${path}/main.py 9155 &
python  ${path}/main.py 9156 &
python  ${path}/main.py 9157 &
python  ${path}/main.py 9158 &
python  ${path}/main.py 9159 &
python  ${path}/main.py 9160 &
python  ${path}/main.py 9161 &
python  ${path}/main.py 9162 &
python  ${path}/main.py 9163 &
python  ${path}/main.py 9164 &
python  ${path}/main.py 9165 &
python  ${path}/main.py 9166 &
python  ${path}/main.py 9167 &
python  ${path}/main.py 9168 &
python  ${path}/main.py 9169 &
python  ${path}/main.py 9170 &
python  ${path}/main.py 9171 &
python  ${path}/main.py 9172 &
python  ${path}/main.py 9173 &
python  ${path}/main.py 9174 &
python  ${path}/main.py 9175 &
python  ${path}/main.py 9176 &
python  ${path}/main.py 9177 &
python  ${path}/main.py 9178 &
python  ${path}/main.py 9179 &
wait
