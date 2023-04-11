#!/bin/bash -l
#SBATCH -J RNN_112
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3996 &
python  ${path}/main.py 3997 &
python  ${path}/main.py 3998 &
python  ${path}/main.py 3999 &
python  ${path}/main.py 4000 &
python  ${path}/main.py 4001 &
python  ${path}/main.py 4002 &
python  ${path}/main.py 4003 &
python  ${path}/main.py 4004 &
python  ${path}/main.py 4005 &
python  ${path}/main.py 4006 &
python  ${path}/main.py 4007 &
python  ${path}/main.py 4008 &
python  ${path}/main.py 4009 &
python  ${path}/main.py 4010 &
python  ${path}/main.py 4011 &
python  ${path}/main.py 4012 &
python  ${path}/main.py 4013 &
python  ${path}/main.py 4014 &
python  ${path}/main.py 4015 &
python  ${path}/main.py 4016 &
python  ${path}/main.py 4017 &
python  ${path}/main.py 4018 &
python  ${path}/main.py 4019 &
python  ${path}/main.py 4020 &
python  ${path}/main.py 4021 &
python  ${path}/main.py 4022 &
python  ${path}/main.py 4023 &
python  ${path}/main.py 4024 &
python  ${path}/main.py 4025 &
python  ${path}/main.py 4026 &
python  ${path}/main.py 4027 &
python  ${path}/main.py 4028 &
python  ${path}/main.py 4029 &
python  ${path}/main.py 4030 &
python  ${path}/main.py 4031 &
wait
