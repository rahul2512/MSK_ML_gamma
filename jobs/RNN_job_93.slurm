#!/bin/bash -l
#SBATCH -J RNN_93
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3312 &
python  ${path}/main.py 3313 &
python  ${path}/main.py 3314 &
python  ${path}/main.py 3315 &
python  ${path}/main.py 3316 &
python  ${path}/main.py 3317 &
python  ${path}/main.py 3318 &
python  ${path}/main.py 3319 &
python  ${path}/main.py 3320 &
python  ${path}/main.py 3321 &
python  ${path}/main.py 3322 &
python  ${path}/main.py 3323 &
python  ${path}/main.py 3324 &
python  ${path}/main.py 3325 &
python  ${path}/main.py 3326 &
python  ${path}/main.py 3327 &
python  ${path}/main.py 3328 &
python  ${path}/main.py 3329 &
python  ${path}/main.py 3330 &
python  ${path}/main.py 3331 &
python  ${path}/main.py 3332 &
python  ${path}/main.py 3333 &
python  ${path}/main.py 3334 &
python  ${path}/main.py 3335 &
python  ${path}/main.py 3336 &
python  ${path}/main.py 3337 &
python  ${path}/main.py 3338 &
python  ${path}/main.py 3339 &
python  ${path}/main.py 3340 &
python  ${path}/main.py 3341 &
python  ${path}/main.py 3342 &
python  ${path}/main.py 3343 &
python  ${path}/main.py 3344 &
python  ${path}/main.py 3345 &
python  ${path}/main.py 3346 &
python  ${path}/main.py 3347 &
wait
