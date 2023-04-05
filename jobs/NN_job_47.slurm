#!/bin/bash -l
#SBATCH -J NN_47
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
python  ${path}/main.py 3348 &
python  ${path}/main.py 3349 &
python  ${path}/main.py 3350 &
python  ${path}/main.py 3351 &
python  ${path}/main.py 3352 &
python  ${path}/main.py 3353 &
python  ${path}/main.py 3354 &
python  ${path}/main.py 3355 &
python  ${path}/main.py 3356 &
python  ${path}/main.py 3357 &
python  ${path}/main.py 3358 &
python  ${path}/main.py 3359 &
python  ${path}/main.py 3360 &
python  ${path}/main.py 3361 &
python  ${path}/main.py 3362 &
python  ${path}/main.py 3363 &
python  ${path}/main.py 3364 &
python  ${path}/main.py 3365 &
python  ${path}/main.py 3366 &
python  ${path}/main.py 3367 &
python  ${path}/main.py 3368 &
python  ${path}/main.py 3369 &
python  ${path}/main.py 3370 &
python  ${path}/main.py 3371 &
python  ${path}/main.py 3372 &
python  ${path}/main.py 3373 &
python  ${path}/main.py 3374 &
python  ${path}/main.py 3375 &
python  ${path}/main.py 3376 &
python  ${path}/main.py 3377 &
python  ${path}/main.py 3378 &
python  ${path}/main.py 3379 &
python  ${path}/main.py 3380 &
python  ${path}/main.py 3381 &
python  ${path}/main.py 3382 &
python  ${path}/main.py 3383 &
wait
