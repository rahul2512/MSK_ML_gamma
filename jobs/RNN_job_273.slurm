#!/bin/bash -l
#SBATCH -J RNN_273
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9792 &
python  ${path}/main.py 9793 &
python  ${path}/main.py 9794 &
python  ${path}/main.py 9795 &
python  ${path}/main.py 9796 &
python  ${path}/main.py 9797 &
python  ${path}/main.py 9798 &
python  ${path}/main.py 9799 &
python  ${path}/main.py 9800 &
python  ${path}/main.py 9801 &
python  ${path}/main.py 9802 &
python  ${path}/main.py 9803 &
python  ${path}/main.py 9804 &
python  ${path}/main.py 9805 &
python  ${path}/main.py 9806 &
python  ${path}/main.py 9807 &
python  ${path}/main.py 9808 &
python  ${path}/main.py 9809 &
python  ${path}/main.py 9810 &
python  ${path}/main.py 9811 &
python  ${path}/main.py 9812 &
python  ${path}/main.py 9813 &
python  ${path}/main.py 9814 &
python  ${path}/main.py 9815 &
python  ${path}/main.py 9816 &
python  ${path}/main.py 9817 &
python  ${path}/main.py 9818 &
python  ${path}/main.py 9819 &
python  ${path}/main.py 9820 &
python  ${path}/main.py 9821 &
python  ${path}/main.py 9822 &
python  ${path}/main.py 9823 &
python  ${path}/main.py 9824 &
python  ${path}/main.py 9825 &
python  ${path}/main.py 9826 &
python  ${path}/main.py 9827 &
wait
