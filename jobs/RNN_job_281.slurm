#!/bin/bash -l
#SBATCH -J RNN_281
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10080 &
python  ${path}/main.py 10081 &
python  ${path}/main.py 10082 &
python  ${path}/main.py 10083 &
python  ${path}/main.py 10084 &
python  ${path}/main.py 10085 &
python  ${path}/main.py 10086 &
python  ${path}/main.py 10087 &
python  ${path}/main.py 10088 &
python  ${path}/main.py 10089 &
python  ${path}/main.py 10090 &
python  ${path}/main.py 10091 &
python  ${path}/main.py 10092 &
python  ${path}/main.py 10093 &
python  ${path}/main.py 10094 &
python  ${path}/main.py 10095 &
python  ${path}/main.py 10096 &
python  ${path}/main.py 10097 &
python  ${path}/main.py 10098 &
python  ${path}/main.py 10099 &
python  ${path}/main.py 10100 &
python  ${path}/main.py 10101 &
python  ${path}/main.py 10102 &
python  ${path}/main.py 10103 &
python  ${path}/main.py 10104 &
python  ${path}/main.py 10105 &
python  ${path}/main.py 10106 &
python  ${path}/main.py 10107 &
python  ${path}/main.py 10108 &
python  ${path}/main.py 10109 &
python  ${path}/main.py 10110 &
python  ${path}/main.py 10111 &
python  ${path}/main.py 10112 &
python  ${path}/main.py 10113 &
python  ${path}/main.py 10114 &
python  ${path}/main.py 10115 &
wait
