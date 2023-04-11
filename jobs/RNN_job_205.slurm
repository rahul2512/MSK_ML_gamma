#!/bin/bash -l
#SBATCH -J RNN_205
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7344 &
python  ${path}/main.py 7345 &
python  ${path}/main.py 7346 &
python  ${path}/main.py 7347 &
python  ${path}/main.py 7348 &
python  ${path}/main.py 7349 &
python  ${path}/main.py 7350 &
python  ${path}/main.py 7351 &
python  ${path}/main.py 7352 &
python  ${path}/main.py 7353 &
python  ${path}/main.py 7354 &
python  ${path}/main.py 7355 &
python  ${path}/main.py 7356 &
python  ${path}/main.py 7357 &
python  ${path}/main.py 7358 &
python  ${path}/main.py 7359 &
python  ${path}/main.py 7360 &
python  ${path}/main.py 7361 &
python  ${path}/main.py 7362 &
python  ${path}/main.py 7363 &
python  ${path}/main.py 7364 &
python  ${path}/main.py 7365 &
python  ${path}/main.py 7366 &
python  ${path}/main.py 7367 &
python  ${path}/main.py 7368 &
python  ${path}/main.py 7369 &
python  ${path}/main.py 7370 &
python  ${path}/main.py 7371 &
python  ${path}/main.py 7372 &
python  ${path}/main.py 7373 &
python  ${path}/main.py 7374 &
python  ${path}/main.py 7375 &
python  ${path}/main.py 7376 &
python  ${path}/main.py 7377 &
python  ${path}/main.py 7378 &
python  ${path}/main.py 7379 &
wait
