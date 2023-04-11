#!/bin/bash -l
#SBATCH -J RNN_107
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3816 &
python  ${path}/main.py 3817 &
python  ${path}/main.py 3818 &
python  ${path}/main.py 3819 &
python  ${path}/main.py 3820 &
python  ${path}/main.py 3821 &
python  ${path}/main.py 3822 &
python  ${path}/main.py 3823 &
python  ${path}/main.py 3824 &
python  ${path}/main.py 3825 &
python  ${path}/main.py 3826 &
python  ${path}/main.py 3827 &
python  ${path}/main.py 3828 &
python  ${path}/main.py 3829 &
python  ${path}/main.py 3830 &
python  ${path}/main.py 3831 &
python  ${path}/main.py 3832 &
python  ${path}/main.py 3833 &
python  ${path}/main.py 3834 &
python  ${path}/main.py 3835 &
python  ${path}/main.py 3836 &
python  ${path}/main.py 3837 &
python  ${path}/main.py 3838 &
python  ${path}/main.py 3839 &
python  ${path}/main.py 3840 &
python  ${path}/main.py 3841 &
python  ${path}/main.py 3842 &
python  ${path}/main.py 3843 &
python  ${path}/main.py 3844 &
python  ${path}/main.py 3845 &
python  ${path}/main.py 3846 &
python  ${path}/main.py 3847 &
python  ${path}/main.py 3848 &
python  ${path}/main.py 3849 &
python  ${path}/main.py 3850 &
python  ${path}/main.py 3851 &
wait
