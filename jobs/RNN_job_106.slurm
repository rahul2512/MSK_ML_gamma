#!/bin/bash -l
#SBATCH -J RNN_106
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3780 &
python  ${path}/main.py 3781 &
python  ${path}/main.py 3782 &
python  ${path}/main.py 3783 &
python  ${path}/main.py 3784 &
python  ${path}/main.py 3785 &
python  ${path}/main.py 3786 &
python  ${path}/main.py 3787 &
python  ${path}/main.py 3788 &
python  ${path}/main.py 3789 &
python  ${path}/main.py 3790 &
python  ${path}/main.py 3791 &
python  ${path}/main.py 3792 &
python  ${path}/main.py 3793 &
python  ${path}/main.py 3794 &
python  ${path}/main.py 3795 &
python  ${path}/main.py 3796 &
python  ${path}/main.py 3797 &
python  ${path}/main.py 3798 &
python  ${path}/main.py 3799 &
python  ${path}/main.py 3800 &
python  ${path}/main.py 3801 &
python  ${path}/main.py 3802 &
python  ${path}/main.py 3803 &
python  ${path}/main.py 3804 &
python  ${path}/main.py 3805 &
python  ${path}/main.py 3806 &
python  ${path}/main.py 3807 &
python  ${path}/main.py 3808 &
python  ${path}/main.py 3809 &
python  ${path}/main.py 3810 &
python  ${path}/main.py 3811 &
python  ${path}/main.py 3812 &
python  ${path}/main.py 3813 &
python  ${path}/main.py 3814 &
python  ${path}/main.py 3815 &
wait
