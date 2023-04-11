#!/bin/bash -l
#SBATCH -J RNN_102
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3636 &
python  ${path}/main.py 3637 &
python  ${path}/main.py 3638 &
python  ${path}/main.py 3639 &
python  ${path}/main.py 3640 &
python  ${path}/main.py 3641 &
python  ${path}/main.py 3642 &
python  ${path}/main.py 3643 &
python  ${path}/main.py 3644 &
python  ${path}/main.py 3645 &
python  ${path}/main.py 3646 &
python  ${path}/main.py 3647 &
python  ${path}/main.py 3648 &
python  ${path}/main.py 3649 &
python  ${path}/main.py 3650 &
python  ${path}/main.py 3651 &
python  ${path}/main.py 3652 &
python  ${path}/main.py 3653 &
python  ${path}/main.py 3654 &
python  ${path}/main.py 3655 &
python  ${path}/main.py 3656 &
python  ${path}/main.py 3657 &
python  ${path}/main.py 3658 &
python  ${path}/main.py 3659 &
python  ${path}/main.py 3660 &
python  ${path}/main.py 3661 &
python  ${path}/main.py 3662 &
python  ${path}/main.py 3663 &
python  ${path}/main.py 3664 &
python  ${path}/main.py 3665 &
python  ${path}/main.py 3666 &
python  ${path}/main.py 3667 &
python  ${path}/main.py 3668 &
python  ${path}/main.py 3669 &
python  ${path}/main.py 3670 &
python  ${path}/main.py 3671 &
wait
