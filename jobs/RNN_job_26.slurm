#!/bin/bash -l
#SBATCH -J RNN_26
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 900 &
python  ${path}/main.py 901 &
python  ${path}/main.py 902 &
python  ${path}/main.py 903 &
python  ${path}/main.py 904 &
python  ${path}/main.py 905 &
python  ${path}/main.py 906 &
python  ${path}/main.py 907 &
python  ${path}/main.py 908 &
python  ${path}/main.py 909 &
python  ${path}/main.py 910 &
python  ${path}/main.py 911 &
python  ${path}/main.py 912 &
python  ${path}/main.py 913 &
python  ${path}/main.py 914 &
python  ${path}/main.py 915 &
python  ${path}/main.py 916 &
python  ${path}/main.py 917 &
python  ${path}/main.py 918 &
python  ${path}/main.py 919 &
python  ${path}/main.py 920 &
python  ${path}/main.py 921 &
python  ${path}/main.py 922 &
python  ${path}/main.py 923 &
python  ${path}/main.py 924 &
python  ${path}/main.py 925 &
python  ${path}/main.py 926 &
python  ${path}/main.py 927 &
python  ${path}/main.py 928 &
python  ${path}/main.py 929 &
python  ${path}/main.py 930 &
python  ${path}/main.py 931 &
python  ${path}/main.py 932 &
python  ${path}/main.py 933 &
python  ${path}/main.py 934 &
python  ${path}/main.py 935 &
wait
