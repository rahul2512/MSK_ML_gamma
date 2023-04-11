#!/bin/bash -l
#SBATCH -J RNN_95
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3384 &
python  ${path}/main.py 3385 &
python  ${path}/main.py 3386 &
python  ${path}/main.py 3387 &
python  ${path}/main.py 3388 &
python  ${path}/main.py 3389 &
python  ${path}/main.py 3390 &
python  ${path}/main.py 3391 &
python  ${path}/main.py 3392 &
python  ${path}/main.py 3393 &
python  ${path}/main.py 3394 &
python  ${path}/main.py 3395 &
python  ${path}/main.py 3396 &
python  ${path}/main.py 3397 &
python  ${path}/main.py 3398 &
python  ${path}/main.py 3399 &
python  ${path}/main.py 3400 &
python  ${path}/main.py 3401 &
python  ${path}/main.py 3402 &
python  ${path}/main.py 3403 &
python  ${path}/main.py 3404 &
python  ${path}/main.py 3405 &
python  ${path}/main.py 3406 &
python  ${path}/main.py 3407 &
python  ${path}/main.py 3408 &
python  ${path}/main.py 3409 &
python  ${path}/main.py 3410 &
python  ${path}/main.py 3411 &
python  ${path}/main.py 3412 &
python  ${path}/main.py 3413 &
python  ${path}/main.py 3414 &
python  ${path}/main.py 3415 &
python  ${path}/main.py 3416 &
python  ${path}/main.py 3417 &
python  ${path}/main.py 3418 &
python  ${path}/main.py 3419 &
wait
