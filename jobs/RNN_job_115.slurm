#!/bin/bash -l
#SBATCH -J RNN_115
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4104 &
python  ${path}/main.py 4105 &
python  ${path}/main.py 4106 &
python  ${path}/main.py 4107 &
python  ${path}/main.py 4108 &
python  ${path}/main.py 4109 &
python  ${path}/main.py 4110 &
python  ${path}/main.py 4111 &
python  ${path}/main.py 4112 &
python  ${path}/main.py 4113 &
python  ${path}/main.py 4114 &
python  ${path}/main.py 4115 &
python  ${path}/main.py 4116 &
python  ${path}/main.py 4117 &
python  ${path}/main.py 4118 &
python  ${path}/main.py 4119 &
python  ${path}/main.py 4120 &
python  ${path}/main.py 4121 &
python  ${path}/main.py 4122 &
python  ${path}/main.py 4123 &
python  ${path}/main.py 4124 &
python  ${path}/main.py 4125 &
python  ${path}/main.py 4126 &
python  ${path}/main.py 4127 &
python  ${path}/main.py 4128 &
python  ${path}/main.py 4129 &
python  ${path}/main.py 4130 &
python  ${path}/main.py 4131 &
python  ${path}/main.py 4132 &
python  ${path}/main.py 4133 &
python  ${path}/main.py 4134 &
python  ${path}/main.py 4135 &
python  ${path}/main.py 4136 &
python  ${path}/main.py 4137 &
python  ${path}/main.py 4138 &
python  ${path}/main.py 4139 &
wait
