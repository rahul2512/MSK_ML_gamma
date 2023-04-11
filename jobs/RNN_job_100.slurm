#!/bin/bash -l
#SBATCH -J RNN_100
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3564 &
python  ${path}/main.py 3565 &
python  ${path}/main.py 3566 &
python  ${path}/main.py 3567 &
python  ${path}/main.py 3568 &
python  ${path}/main.py 3569 &
python  ${path}/main.py 3570 &
python  ${path}/main.py 3571 &
python  ${path}/main.py 3572 &
python  ${path}/main.py 3573 &
python  ${path}/main.py 3574 &
python  ${path}/main.py 3575 &
python  ${path}/main.py 3576 &
python  ${path}/main.py 3577 &
python  ${path}/main.py 3578 &
python  ${path}/main.py 3579 &
python  ${path}/main.py 3580 &
python  ${path}/main.py 3581 &
python  ${path}/main.py 3582 &
python  ${path}/main.py 3583 &
python  ${path}/main.py 3584 &
python  ${path}/main.py 3585 &
python  ${path}/main.py 3586 &
python  ${path}/main.py 3587 &
python  ${path}/main.py 3588 &
python  ${path}/main.py 3589 &
python  ${path}/main.py 3590 &
python  ${path}/main.py 3591 &
python  ${path}/main.py 3592 &
python  ${path}/main.py 3593 &
python  ${path}/main.py 3594 &
python  ${path}/main.py 3595 &
python  ${path}/main.py 3596 &
python  ${path}/main.py 3597 &
python  ${path}/main.py 3598 &
python  ${path}/main.py 3599 &
wait
