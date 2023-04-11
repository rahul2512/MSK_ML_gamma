#!/bin/bash -l
#SBATCH -J RNN_92
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3276 &
python  ${path}/main.py 3277 &
python  ${path}/main.py 3278 &
python  ${path}/main.py 3279 &
python  ${path}/main.py 3280 &
python  ${path}/main.py 3281 &
python  ${path}/main.py 3282 &
python  ${path}/main.py 3283 &
python  ${path}/main.py 3284 &
python  ${path}/main.py 3285 &
python  ${path}/main.py 3286 &
python  ${path}/main.py 3287 &
python  ${path}/main.py 3288 &
python  ${path}/main.py 3289 &
python  ${path}/main.py 3290 &
python  ${path}/main.py 3291 &
python  ${path}/main.py 3292 &
python  ${path}/main.py 3293 &
python  ${path}/main.py 3294 &
python  ${path}/main.py 3295 &
python  ${path}/main.py 3296 &
python  ${path}/main.py 3297 &
python  ${path}/main.py 3298 &
python  ${path}/main.py 3299 &
python  ${path}/main.py 3300 &
python  ${path}/main.py 3301 &
python  ${path}/main.py 3302 &
python  ${path}/main.py 3303 &
python  ${path}/main.py 3304 &
python  ${path}/main.py 3305 &
python  ${path}/main.py 3306 &
python  ${path}/main.py 3307 &
python  ${path}/main.py 3308 &
python  ${path}/main.py 3309 &
python  ${path}/main.py 3310 &
python  ${path}/main.py 3311 &
wait
