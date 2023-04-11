#!/bin/bash -l
#SBATCH -J RNN_274
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9828 &
python  ${path}/main.py 9829 &
python  ${path}/main.py 9830 &
python  ${path}/main.py 9831 &
python  ${path}/main.py 9832 &
python  ${path}/main.py 9833 &
python  ${path}/main.py 9834 &
python  ${path}/main.py 9835 &
python  ${path}/main.py 9836 &
python  ${path}/main.py 9837 &
python  ${path}/main.py 9838 &
python  ${path}/main.py 9839 &
python  ${path}/main.py 9840 &
python  ${path}/main.py 9841 &
python  ${path}/main.py 9842 &
python  ${path}/main.py 9843 &
python  ${path}/main.py 9844 &
python  ${path}/main.py 9845 &
python  ${path}/main.py 9846 &
python  ${path}/main.py 9847 &
python  ${path}/main.py 9848 &
python  ${path}/main.py 9849 &
python  ${path}/main.py 9850 &
python  ${path}/main.py 9851 &
python  ${path}/main.py 9852 &
python  ${path}/main.py 9853 &
python  ${path}/main.py 9854 &
python  ${path}/main.py 9855 &
python  ${path}/main.py 9856 &
python  ${path}/main.py 9857 &
python  ${path}/main.py 9858 &
python  ${path}/main.py 9859 &
python  ${path}/main.py 9860 &
python  ${path}/main.py 9861 &
python  ${path}/main.py 9862 &
python  ${path}/main.py 9863 &
wait
