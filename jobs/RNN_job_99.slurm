#!/bin/bash -l
#SBATCH -J RNN_99
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3528 &
python  ${path}/main.py 3529 &
python  ${path}/main.py 3530 &
python  ${path}/main.py 3531 &
python  ${path}/main.py 3532 &
python  ${path}/main.py 3533 &
python  ${path}/main.py 3534 &
python  ${path}/main.py 3535 &
python  ${path}/main.py 3536 &
python  ${path}/main.py 3537 &
python  ${path}/main.py 3538 &
python  ${path}/main.py 3539 &
python  ${path}/main.py 3540 &
python  ${path}/main.py 3541 &
python  ${path}/main.py 3542 &
python  ${path}/main.py 3543 &
python  ${path}/main.py 3544 &
python  ${path}/main.py 3545 &
python  ${path}/main.py 3546 &
python  ${path}/main.py 3547 &
python  ${path}/main.py 3548 &
python  ${path}/main.py 3549 &
python  ${path}/main.py 3550 &
python  ${path}/main.py 3551 &
python  ${path}/main.py 3552 &
python  ${path}/main.py 3553 &
python  ${path}/main.py 3554 &
python  ${path}/main.py 3555 &
python  ${path}/main.py 3556 &
python  ${path}/main.py 3557 &
python  ${path}/main.py 3558 &
python  ${path}/main.py 3559 &
python  ${path}/main.py 3560 &
python  ${path}/main.py 3561 &
python  ${path}/main.py 3562 &
python  ${path}/main.py 3563 &
wait
