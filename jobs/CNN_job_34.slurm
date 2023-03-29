#!/bin/bash -l
#SBATCH -J CNN_34
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 792 &
python  ${path}/main.py 793 &
python  ${path}/main.py 794 &
python  ${path}/main.py 795 &
python  ${path}/main.py 796 &
python  ${path}/main.py 797 &
python  ${path}/main.py 798 &
python  ${path}/main.py 799 &
python  ${path}/main.py 800 &
python  ${path}/main.py 801 &
python  ${path}/main.py 802 &
python  ${path}/main.py 803 &
python  ${path}/main.py 804 &
python  ${path}/main.py 805 &
python  ${path}/main.py 806 &
python  ${path}/main.py 807 &
python  ${path}/main.py 808 &
python  ${path}/main.py 809 &
python  ${path}/main.py 810 &
python  ${path}/main.py 811 &
python  ${path}/main.py 812 &
python  ${path}/main.py 813 &
python  ${path}/main.py 814 &
python  ${path}/main.py 815 &
wait
