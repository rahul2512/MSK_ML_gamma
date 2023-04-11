#!/bin/bash -l
#SBATCH -J RNN_113
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4032 &
python  ${path}/main.py 4033 &
python  ${path}/main.py 4034 &
python  ${path}/main.py 4035 &
python  ${path}/main.py 4036 &
python  ${path}/main.py 4037 &
python  ${path}/main.py 4038 &
python  ${path}/main.py 4039 &
python  ${path}/main.py 4040 &
python  ${path}/main.py 4041 &
python  ${path}/main.py 4042 &
python  ${path}/main.py 4043 &
python  ${path}/main.py 4044 &
python  ${path}/main.py 4045 &
python  ${path}/main.py 4046 &
python  ${path}/main.py 4047 &
python  ${path}/main.py 4048 &
python  ${path}/main.py 4049 &
python  ${path}/main.py 4050 &
python  ${path}/main.py 4051 &
python  ${path}/main.py 4052 &
python  ${path}/main.py 4053 &
python  ${path}/main.py 4054 &
python  ${path}/main.py 4055 &
python  ${path}/main.py 4056 &
python  ${path}/main.py 4057 &
python  ${path}/main.py 4058 &
python  ${path}/main.py 4059 &
python  ${path}/main.py 4060 &
python  ${path}/main.py 4061 &
python  ${path}/main.py 4062 &
python  ${path}/main.py 4063 &
python  ${path}/main.py 4064 &
python  ${path}/main.py 4065 &
python  ${path}/main.py 4066 &
python  ${path}/main.py 4067 &
wait
