#!/bin/bash -l
#SBATCH -J RNN_248
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8892 &
python  ${path}/main.py 8893 &
python  ${path}/main.py 8894 &
python  ${path}/main.py 8895 &
python  ${path}/main.py 8896 &
python  ${path}/main.py 8897 &
python  ${path}/main.py 8898 &
python  ${path}/main.py 8899 &
python  ${path}/main.py 8900 &
python  ${path}/main.py 8901 &
python  ${path}/main.py 8902 &
python  ${path}/main.py 8903 &
python  ${path}/main.py 8904 &
python  ${path}/main.py 8905 &
python  ${path}/main.py 8906 &
python  ${path}/main.py 8907 &
python  ${path}/main.py 8908 &
python  ${path}/main.py 8909 &
python  ${path}/main.py 8910 &
python  ${path}/main.py 8911 &
python  ${path}/main.py 8912 &
python  ${path}/main.py 8913 &
python  ${path}/main.py 8914 &
python  ${path}/main.py 8915 &
python  ${path}/main.py 8916 &
python  ${path}/main.py 8917 &
python  ${path}/main.py 8918 &
python  ${path}/main.py 8919 &
python  ${path}/main.py 8920 &
python  ${path}/main.py 8921 &
python  ${path}/main.py 8922 &
python  ${path}/main.py 8923 &
python  ${path}/main.py 8924 &
python  ${path}/main.py 8925 &
python  ${path}/main.py 8926 &
python  ${path}/main.py 8927 &
wait
