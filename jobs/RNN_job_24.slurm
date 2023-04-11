#!/bin/bash -l
#SBATCH -J RNN_24
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
python  ${path}/main.py 840 &
python  ${path}/main.py 841 &
python  ${path}/main.py 842 &
python  ${path}/main.py 843 &
python  ${path}/main.py 844 &
python  ${path}/main.py 845 &
python  ${path}/main.py 846 &
python  ${path}/main.py 847 &
python  ${path}/main.py 848 &
python  ${path}/main.py 849 &
python  ${path}/main.py 850 &
python  ${path}/main.py 851 &
python  ${path}/main.py 852 &
python  ${path}/main.py 853 &
python  ${path}/main.py 854 &
python  ${path}/main.py 855 &
python  ${path}/main.py 856 &
python  ${path}/main.py 857 &
python  ${path}/main.py 858 &
python  ${path}/main.py 859 &
python  ${path}/main.py 860 &
python  ${path}/main.py 861 &
python  ${path}/main.py 862 &
python  ${path}/main.py 863 &
wait
