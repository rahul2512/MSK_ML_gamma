#!/bin/bash -l
#SBATCH -J NN_58
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
python  ${path}/main.py 4140 &
python  ${path}/main.py 4141 &
python  ${path}/main.py 4142 &
python  ${path}/main.py 4143 &
python  ${path}/main.py 4144 &
python  ${path}/main.py 4145 &
python  ${path}/main.py 4146 &
python  ${path}/main.py 4147 &
python  ${path}/main.py 4148 &
python  ${path}/main.py 4149 &
python  ${path}/main.py 4150 &
python  ${path}/main.py 4151 &
python  ${path}/main.py 4152 &
python  ${path}/main.py 4153 &
python  ${path}/main.py 4154 &
python  ${path}/main.py 4155 &
python  ${path}/main.py 4156 &
python  ${path}/main.py 4157 &
python  ${path}/main.py 4158 &
python  ${path}/main.py 4159 &
python  ${path}/main.py 4160 &
python  ${path}/main.py 4161 &
python  ${path}/main.py 4162 &
python  ${path}/main.py 4163 &
python  ${path}/main.py 4164 &
python  ${path}/main.py 4165 &
python  ${path}/main.py 4166 &
python  ${path}/main.py 4167 &
python  ${path}/main.py 4168 &
python  ${path}/main.py 4169 &
python  ${path}/main.py 4170 &
python  ${path}/main.py 4171 &
python  ${path}/main.py 4172 &
python  ${path}/main.py 4173 &
python  ${path}/main.py 4174 &
python  ${path}/main.py 4175 &
wait
