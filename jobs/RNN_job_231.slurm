#!/bin/bash -l
#SBATCH -J RNN_231
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8280 &
python  ${path}/main.py 8281 &
python  ${path}/main.py 8282 &
python  ${path}/main.py 8283 &
python  ${path}/main.py 8284 &
python  ${path}/main.py 8285 &
python  ${path}/main.py 8286 &
python  ${path}/main.py 8287 &
python  ${path}/main.py 8288 &
python  ${path}/main.py 8289 &
python  ${path}/main.py 8290 &
python  ${path}/main.py 8291 &
python  ${path}/main.py 8292 &
python  ${path}/main.py 8293 &
python  ${path}/main.py 8294 &
python  ${path}/main.py 8295 &
python  ${path}/main.py 8296 &
python  ${path}/main.py 8297 &
python  ${path}/main.py 8298 &
python  ${path}/main.py 8299 &
python  ${path}/main.py 8300 &
python  ${path}/main.py 8301 &
python  ${path}/main.py 8302 &
python  ${path}/main.py 8303 &
python  ${path}/main.py 8304 &
python  ${path}/main.py 8305 &
python  ${path}/main.py 8306 &
python  ${path}/main.py 8307 &
python  ${path}/main.py 8308 &
python  ${path}/main.py 8309 &
python  ${path}/main.py 8310 &
python  ${path}/main.py 8311 &
python  ${path}/main.py 8312 &
python  ${path}/main.py 8313 &
python  ${path}/main.py 8314 &
python  ${path}/main.py 8315 &
wait
