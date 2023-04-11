#!/bin/bash -l
#SBATCH -J RNN_193
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6912 &
python  ${path}/main.py 6913 &
python  ${path}/main.py 6914 &
python  ${path}/main.py 6915 &
python  ${path}/main.py 6916 &
python  ${path}/main.py 6917 &
python  ${path}/main.py 6918 &
python  ${path}/main.py 6919 &
python  ${path}/main.py 6920 &
python  ${path}/main.py 6921 &
python  ${path}/main.py 6922 &
python  ${path}/main.py 6923 &
python  ${path}/main.py 6924 &
python  ${path}/main.py 6925 &
python  ${path}/main.py 6926 &
python  ${path}/main.py 6927 &
python  ${path}/main.py 6928 &
python  ${path}/main.py 6929 &
python  ${path}/main.py 6930 &
python  ${path}/main.py 6931 &
python  ${path}/main.py 6932 &
python  ${path}/main.py 6933 &
python  ${path}/main.py 6934 &
python  ${path}/main.py 6935 &
python  ${path}/main.py 6936 &
python  ${path}/main.py 6937 &
python  ${path}/main.py 6938 &
python  ${path}/main.py 6939 &
python  ${path}/main.py 6940 &
python  ${path}/main.py 6941 &
python  ${path}/main.py 6942 &
python  ${path}/main.py 6943 &
python  ${path}/main.py 6944 &
python  ${path}/main.py 6945 &
python  ${path}/main.py 6946 &
python  ${path}/main.py 6947 &
wait
