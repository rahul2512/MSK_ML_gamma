#!/bin/bash -l
#SBATCH -J RNN_114
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4068 &
python  ${path}/main.py 4069 &
python  ${path}/main.py 4070 &
python  ${path}/main.py 4071 &
python  ${path}/main.py 4072 &
python  ${path}/main.py 4073 &
python  ${path}/main.py 4074 &
python  ${path}/main.py 4075 &
python  ${path}/main.py 4076 &
python  ${path}/main.py 4077 &
python  ${path}/main.py 4078 &
python  ${path}/main.py 4079 &
python  ${path}/main.py 4080 &
python  ${path}/main.py 4081 &
python  ${path}/main.py 4082 &
python  ${path}/main.py 4083 &
python  ${path}/main.py 4084 &
python  ${path}/main.py 4085 &
python  ${path}/main.py 4086 &
python  ${path}/main.py 4087 &
python  ${path}/main.py 4088 &
python  ${path}/main.py 4089 &
python  ${path}/main.py 4090 &
python  ${path}/main.py 4091 &
python  ${path}/main.py 4092 &
python  ${path}/main.py 4093 &
python  ${path}/main.py 4094 &
python  ${path}/main.py 4095 &
python  ${path}/main.py 4096 &
python  ${path}/main.py 4097 &
python  ${path}/main.py 4098 &
python  ${path}/main.py 4099 &
python  ${path}/main.py 4100 &
python  ${path}/main.py 4101 &
python  ${path}/main.py 4102 &
python  ${path}/main.py 4103 &
wait
