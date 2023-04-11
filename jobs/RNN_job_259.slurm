#!/bin/bash -l
#SBATCH -J RNN_259
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9288 &
python  ${path}/main.py 9289 &
python  ${path}/main.py 9290 &
python  ${path}/main.py 9291 &
python  ${path}/main.py 9292 &
python  ${path}/main.py 9293 &
python  ${path}/main.py 9294 &
python  ${path}/main.py 9295 &
python  ${path}/main.py 9296 &
python  ${path}/main.py 9297 &
python  ${path}/main.py 9298 &
python  ${path}/main.py 9299 &
python  ${path}/main.py 9300 &
python  ${path}/main.py 9301 &
python  ${path}/main.py 9302 &
python  ${path}/main.py 9303 &
python  ${path}/main.py 9304 &
python  ${path}/main.py 9305 &
python  ${path}/main.py 9306 &
python  ${path}/main.py 9307 &
python  ${path}/main.py 9308 &
python  ${path}/main.py 9309 &
python  ${path}/main.py 9310 &
python  ${path}/main.py 9311 &
python  ${path}/main.py 9312 &
python  ${path}/main.py 9313 &
python  ${path}/main.py 9314 &
python  ${path}/main.py 9315 &
python  ${path}/main.py 9316 &
python  ${path}/main.py 9317 &
python  ${path}/main.py 9318 &
python  ${path}/main.py 9319 &
python  ${path}/main.py 9320 &
python  ${path}/main.py 9321 &
python  ${path}/main.py 9322 &
python  ${path}/main.py 9323 &
wait
