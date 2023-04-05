#!/bin/bash -l
#SBATCH -J NN_57
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
