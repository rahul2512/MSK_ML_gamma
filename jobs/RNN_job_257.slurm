#!/bin/bash -l
#SBATCH -J RNN_257
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9216 &
python  ${path}/main.py 9217 &
python  ${path}/main.py 9218 &
python  ${path}/main.py 9219 &
python  ${path}/main.py 9220 &
python  ${path}/main.py 9221 &
python  ${path}/main.py 9222 &
python  ${path}/main.py 9223 &
python  ${path}/main.py 9224 &
python  ${path}/main.py 9225 &
python  ${path}/main.py 9226 &
python  ${path}/main.py 9227 &
python  ${path}/main.py 9228 &
python  ${path}/main.py 9229 &
python  ${path}/main.py 9230 &
python  ${path}/main.py 9231 &
python  ${path}/main.py 9232 &
python  ${path}/main.py 9233 &
python  ${path}/main.py 9234 &
python  ${path}/main.py 9235 &
python  ${path}/main.py 9236 &
python  ${path}/main.py 9237 &
python  ${path}/main.py 9238 &
python  ${path}/main.py 9239 &
python  ${path}/main.py 9240 &
python  ${path}/main.py 9241 &
python  ${path}/main.py 9242 &
python  ${path}/main.py 9243 &
python  ${path}/main.py 9244 &
python  ${path}/main.py 9245 &
python  ${path}/main.py 9246 &
python  ${path}/main.py 9247 &
python  ${path}/main.py 9248 &
python  ${path}/main.py 9249 &
python  ${path}/main.py 9250 &
python  ${path}/main.py 9251 &
wait
