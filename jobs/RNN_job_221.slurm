#!/bin/bash -l
#SBATCH -J RNN_221
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7920 &
python  ${path}/main.py 7921 &
python  ${path}/main.py 7922 &
python  ${path}/main.py 7923 &
python  ${path}/main.py 7924 &
python  ${path}/main.py 7925 &
python  ${path}/main.py 7926 &
python  ${path}/main.py 7927 &
python  ${path}/main.py 7928 &
python  ${path}/main.py 7929 &
python  ${path}/main.py 7930 &
python  ${path}/main.py 7931 &
python  ${path}/main.py 7932 &
python  ${path}/main.py 7933 &
python  ${path}/main.py 7934 &
python  ${path}/main.py 7935 &
python  ${path}/main.py 7936 &
python  ${path}/main.py 7937 &
python  ${path}/main.py 7938 &
python  ${path}/main.py 7939 &
python  ${path}/main.py 7940 &
python  ${path}/main.py 7941 &
python  ${path}/main.py 7942 &
python  ${path}/main.py 7943 &
python  ${path}/main.py 7944 &
python  ${path}/main.py 7945 &
python  ${path}/main.py 7946 &
python  ${path}/main.py 7947 &
python  ${path}/main.py 7948 &
python  ${path}/main.py 7949 &
python  ${path}/main.py 7950 &
python  ${path}/main.py 7951 &
python  ${path}/main.py 7952 &
python  ${path}/main.py 7953 &
python  ${path}/main.py 7954 &
python  ${path}/main.py 7955 &
wait
