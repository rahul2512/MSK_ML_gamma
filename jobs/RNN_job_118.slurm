#!/bin/bash -l
#SBATCH -J RNN_118
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4212 &
python  ${path}/main.py 4213 &
python  ${path}/main.py 4214 &
python  ${path}/main.py 4215 &
python  ${path}/main.py 4216 &
python  ${path}/main.py 4217 &
python  ${path}/main.py 4218 &
python  ${path}/main.py 4219 &
python  ${path}/main.py 4220 &
python  ${path}/main.py 4221 &
python  ${path}/main.py 4222 &
python  ${path}/main.py 4223 &
python  ${path}/main.py 4224 &
python  ${path}/main.py 4225 &
python  ${path}/main.py 4226 &
python  ${path}/main.py 4227 &
python  ${path}/main.py 4228 &
python  ${path}/main.py 4229 &
python  ${path}/main.py 4230 &
python  ${path}/main.py 4231 &
python  ${path}/main.py 4232 &
python  ${path}/main.py 4233 &
python  ${path}/main.py 4234 &
python  ${path}/main.py 4235 &
python  ${path}/main.py 4236 &
python  ${path}/main.py 4237 &
python  ${path}/main.py 4238 &
python  ${path}/main.py 4239 &
python  ${path}/main.py 4240 &
python  ${path}/main.py 4241 &
python  ${path}/main.py 4242 &
python  ${path}/main.py 4243 &
python  ${path}/main.py 4244 &
python  ${path}/main.py 4245 &
python  ${path}/main.py 4246 &
python  ${path}/main.py 4247 &
wait
