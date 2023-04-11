#!/bin/bash -l
#SBATCH -J RNN_97
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3456 &
python  ${path}/main.py 3457 &
python  ${path}/main.py 3458 &
python  ${path}/main.py 3459 &
python  ${path}/main.py 3460 &
python  ${path}/main.py 3461 &
python  ${path}/main.py 3462 &
python  ${path}/main.py 3463 &
python  ${path}/main.py 3464 &
python  ${path}/main.py 3465 &
python  ${path}/main.py 3466 &
python  ${path}/main.py 3467 &
python  ${path}/main.py 3468 &
python  ${path}/main.py 3469 &
python  ${path}/main.py 3470 &
python  ${path}/main.py 3471 &
python  ${path}/main.py 3472 &
python  ${path}/main.py 3473 &
python  ${path}/main.py 3474 &
python  ${path}/main.py 3475 &
python  ${path}/main.py 3476 &
python  ${path}/main.py 3477 &
python  ${path}/main.py 3478 &
python  ${path}/main.py 3479 &
python  ${path}/main.py 3480 &
python  ${path}/main.py 3481 &
python  ${path}/main.py 3482 &
python  ${path}/main.py 3483 &
python  ${path}/main.py 3484 &
python  ${path}/main.py 3485 &
python  ${path}/main.py 3486 &
python  ${path}/main.py 3487 &
python  ${path}/main.py 3488 &
python  ${path}/main.py 3489 &
python  ${path}/main.py 3490 &
python  ${path}/main.py 3491 &
wait
