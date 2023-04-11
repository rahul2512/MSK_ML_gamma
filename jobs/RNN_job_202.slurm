#!/bin/bash -l
#SBATCH -J RNN_202
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7236 &
python  ${path}/main.py 7237 &
python  ${path}/main.py 7238 &
python  ${path}/main.py 7239 &
python  ${path}/main.py 7240 &
python  ${path}/main.py 7241 &
python  ${path}/main.py 7242 &
python  ${path}/main.py 7243 &
python  ${path}/main.py 7244 &
python  ${path}/main.py 7245 &
python  ${path}/main.py 7246 &
python  ${path}/main.py 7247 &
python  ${path}/main.py 7248 &
python  ${path}/main.py 7249 &
python  ${path}/main.py 7250 &
python  ${path}/main.py 7251 &
python  ${path}/main.py 7252 &
python  ${path}/main.py 7253 &
python  ${path}/main.py 7254 &
python  ${path}/main.py 7255 &
python  ${path}/main.py 7256 &
python  ${path}/main.py 7257 &
python  ${path}/main.py 7258 &
python  ${path}/main.py 7259 &
python  ${path}/main.py 7260 &
python  ${path}/main.py 7261 &
python  ${path}/main.py 7262 &
python  ${path}/main.py 7263 &
python  ${path}/main.py 7264 &
python  ${path}/main.py 7265 &
python  ${path}/main.py 7266 &
python  ${path}/main.py 7267 &
python  ${path}/main.py 7268 &
python  ${path}/main.py 7269 &
python  ${path}/main.py 7270 &
python  ${path}/main.py 7271 &
wait
