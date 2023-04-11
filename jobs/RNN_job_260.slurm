#!/bin/bash -l
#SBATCH -J RNN_260
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9324 &
python  ${path}/main.py 9325 &
python  ${path}/main.py 9326 &
python  ${path}/main.py 9327 &
python  ${path}/main.py 9328 &
python  ${path}/main.py 9329 &
python  ${path}/main.py 9330 &
python  ${path}/main.py 9331 &
python  ${path}/main.py 9332 &
python  ${path}/main.py 9333 &
python  ${path}/main.py 9334 &
python  ${path}/main.py 9335 &
python  ${path}/main.py 9336 &
python  ${path}/main.py 9337 &
python  ${path}/main.py 9338 &
python  ${path}/main.py 9339 &
python  ${path}/main.py 9340 &
python  ${path}/main.py 9341 &
python  ${path}/main.py 9342 &
python  ${path}/main.py 9343 &
python  ${path}/main.py 9344 &
python  ${path}/main.py 9345 &
python  ${path}/main.py 9346 &
python  ${path}/main.py 9347 &
python  ${path}/main.py 9348 &
python  ${path}/main.py 9349 &
python  ${path}/main.py 9350 &
python  ${path}/main.py 9351 &
python  ${path}/main.py 9352 &
python  ${path}/main.py 9353 &
python  ${path}/main.py 9354 &
python  ${path}/main.py 9355 &
python  ${path}/main.py 9356 &
python  ${path}/main.py 9357 &
python  ${path}/main.py 9358 &
python  ${path}/main.py 9359 &
wait
