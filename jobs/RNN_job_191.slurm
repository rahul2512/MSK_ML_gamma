#!/bin/bash -l
#SBATCH -J RNN_191
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6840 &
python  ${path}/main.py 6841 &
python  ${path}/main.py 6842 &
python  ${path}/main.py 6843 &
python  ${path}/main.py 6844 &
python  ${path}/main.py 6845 &
python  ${path}/main.py 6846 &
python  ${path}/main.py 6847 &
python  ${path}/main.py 6848 &
python  ${path}/main.py 6849 &
python  ${path}/main.py 6850 &
python  ${path}/main.py 6851 &
python  ${path}/main.py 6852 &
python  ${path}/main.py 6853 &
python  ${path}/main.py 6854 &
python  ${path}/main.py 6855 &
python  ${path}/main.py 6856 &
python  ${path}/main.py 6857 &
python  ${path}/main.py 6858 &
python  ${path}/main.py 6859 &
python  ${path}/main.py 6860 &
python  ${path}/main.py 6861 &
python  ${path}/main.py 6862 &
python  ${path}/main.py 6863 &
python  ${path}/main.py 6864 &
python  ${path}/main.py 6865 &
python  ${path}/main.py 6866 &
python  ${path}/main.py 6867 &
python  ${path}/main.py 6868 &
python  ${path}/main.py 6869 &
python  ${path}/main.py 6870 &
python  ${path}/main.py 6871 &
python  ${path}/main.py 6872 &
python  ${path}/main.py 6873 &
python  ${path}/main.py 6874 &
python  ${path}/main.py 6875 &
wait
