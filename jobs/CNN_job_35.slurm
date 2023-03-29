#!/bin/bash -l
#SBATCH -J CNN_35
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 816 &
python  ${path}/main.py 817 &
python  ${path}/main.py 818 &
python  ${path}/main.py 819 &
python  ${path}/main.py 820 &
python  ${path}/main.py 821 &
python  ${path}/main.py 822 &
python  ${path}/main.py 823 &
python  ${path}/main.py 824 &
python  ${path}/main.py 825 &
python  ${path}/main.py 826 &
python  ${path}/main.py 827 &
python  ${path}/main.py 828 &
python  ${path}/main.py 829 &
python  ${path}/main.py 830 &
python  ${path}/main.py 831 &
python  ${path}/main.py 832 &
python  ${path}/main.py 833 &
python  ${path}/main.py 834 &
python  ${path}/main.py 835 &
python  ${path}/main.py 836 &
python  ${path}/main.py 837 &
python  ${path}/main.py 838 &
python  ${path}/main.py 839 &
wait
