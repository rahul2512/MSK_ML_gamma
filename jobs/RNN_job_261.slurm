#!/bin/bash -l
#SBATCH -J RNN_261
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9360 &
python  ${path}/main.py 9361 &
python  ${path}/main.py 9362 &
python  ${path}/main.py 9363 &
python  ${path}/main.py 9364 &
python  ${path}/main.py 9365 &
python  ${path}/main.py 9366 &
python  ${path}/main.py 9367 &
python  ${path}/main.py 9368 &
python  ${path}/main.py 9369 &
python  ${path}/main.py 9370 &
python  ${path}/main.py 9371 &
python  ${path}/main.py 9372 &
python  ${path}/main.py 9373 &
python  ${path}/main.py 9374 &
python  ${path}/main.py 9375 &
python  ${path}/main.py 9376 &
python  ${path}/main.py 9377 &
python  ${path}/main.py 9378 &
python  ${path}/main.py 9379 &
python  ${path}/main.py 9380 &
python  ${path}/main.py 9381 &
python  ${path}/main.py 9382 &
python  ${path}/main.py 9383 &
python  ${path}/main.py 9384 &
python  ${path}/main.py 9385 &
python  ${path}/main.py 9386 &
python  ${path}/main.py 9387 &
python  ${path}/main.py 9388 &
python  ${path}/main.py 9389 &
python  ${path}/main.py 9390 &
python  ${path}/main.py 9391 &
python  ${path}/main.py 9392 &
python  ${path}/main.py 9393 &
python  ${path}/main.py 9394 &
python  ${path}/main.py 9395 &
wait
