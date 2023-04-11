#!/bin/bash -l
#SBATCH -J RNN_104
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3708 &
python  ${path}/main.py 3709 &
python  ${path}/main.py 3710 &
python  ${path}/main.py 3711 &
python  ${path}/main.py 3712 &
python  ${path}/main.py 3713 &
python  ${path}/main.py 3714 &
python  ${path}/main.py 3715 &
python  ${path}/main.py 3716 &
python  ${path}/main.py 3717 &
python  ${path}/main.py 3718 &
python  ${path}/main.py 3719 &
python  ${path}/main.py 3720 &
python  ${path}/main.py 3721 &
python  ${path}/main.py 3722 &
python  ${path}/main.py 3723 &
python  ${path}/main.py 3724 &
python  ${path}/main.py 3725 &
python  ${path}/main.py 3726 &
python  ${path}/main.py 3727 &
python  ${path}/main.py 3728 &
python  ${path}/main.py 3729 &
python  ${path}/main.py 3730 &
python  ${path}/main.py 3731 &
python  ${path}/main.py 3732 &
python  ${path}/main.py 3733 &
python  ${path}/main.py 3734 &
python  ${path}/main.py 3735 &
python  ${path}/main.py 3736 &
python  ${path}/main.py 3737 &
python  ${path}/main.py 3738 &
python  ${path}/main.py 3739 &
python  ${path}/main.py 3740 &
python  ${path}/main.py 3741 &
python  ${path}/main.py 3742 &
python  ${path}/main.py 3743 &
wait
