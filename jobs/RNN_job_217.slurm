#!/bin/bash -l
#SBATCH -J RNN_217
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7776 &
python  ${path}/main.py 7777 &
python  ${path}/main.py 7778 &
python  ${path}/main.py 7779 &
python  ${path}/main.py 7780 &
python  ${path}/main.py 7781 &
python  ${path}/main.py 7782 &
python  ${path}/main.py 7783 &
python  ${path}/main.py 7784 &
python  ${path}/main.py 7785 &
python  ${path}/main.py 7786 &
python  ${path}/main.py 7787 &
python  ${path}/main.py 7788 &
python  ${path}/main.py 7789 &
python  ${path}/main.py 7790 &
python  ${path}/main.py 7791 &
python  ${path}/main.py 7792 &
python  ${path}/main.py 7793 &
python  ${path}/main.py 7794 &
python  ${path}/main.py 7795 &
python  ${path}/main.py 7796 &
python  ${path}/main.py 7797 &
python  ${path}/main.py 7798 &
python  ${path}/main.py 7799 &
python  ${path}/main.py 7800 &
python  ${path}/main.py 7801 &
python  ${path}/main.py 7802 &
python  ${path}/main.py 7803 &
python  ${path}/main.py 7804 &
python  ${path}/main.py 7805 &
python  ${path}/main.py 7806 &
python  ${path}/main.py 7807 &
python  ${path}/main.py 7808 &
python  ${path}/main.py 7809 &
python  ${path}/main.py 7810 &
python  ${path}/main.py 7811 &
wait
