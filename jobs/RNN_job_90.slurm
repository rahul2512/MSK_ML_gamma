#!/bin/bash -l
#SBATCH -J RNN_90
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3204 &
python  ${path}/main.py 3205 &
python  ${path}/main.py 3206 &
python  ${path}/main.py 3207 &
python  ${path}/main.py 3208 &
python  ${path}/main.py 3209 &
python  ${path}/main.py 3210 &
python  ${path}/main.py 3211 &
python  ${path}/main.py 3212 &
python  ${path}/main.py 3213 &
python  ${path}/main.py 3214 &
python  ${path}/main.py 3215 &
python  ${path}/main.py 3216 &
python  ${path}/main.py 3217 &
python  ${path}/main.py 3218 &
python  ${path}/main.py 3219 &
python  ${path}/main.py 3220 &
python  ${path}/main.py 3221 &
python  ${path}/main.py 3222 &
python  ${path}/main.py 3223 &
python  ${path}/main.py 3224 &
python  ${path}/main.py 3225 &
python  ${path}/main.py 3226 &
python  ${path}/main.py 3227 &
python  ${path}/main.py 3228 &
python  ${path}/main.py 3229 &
python  ${path}/main.py 3230 &
python  ${path}/main.py 3231 &
python  ${path}/main.py 3232 &
python  ${path}/main.py 3233 &
python  ${path}/main.py 3234 &
python  ${path}/main.py 3235 &
python  ${path}/main.py 3236 &
python  ${path}/main.py 3237 &
python  ${path}/main.py 3238 &
python  ${path}/main.py 3239 &
wait
