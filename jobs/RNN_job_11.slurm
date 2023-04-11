#!/bin/bash -l
#SBATCH -J RNN_11
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 360 &
python  ${path}/main.py 361 &
python  ${path}/main.py 362 &
python  ${path}/main.py 363 &
python  ${path}/main.py 364 &
python  ${path}/main.py 365 &
python  ${path}/main.py 366 &
python  ${path}/main.py 367 &
python  ${path}/main.py 368 &
python  ${path}/main.py 369 &
python  ${path}/main.py 370 &
python  ${path}/main.py 371 &
python  ${path}/main.py 372 &
python  ${path}/main.py 373 &
python  ${path}/main.py 374 &
python  ${path}/main.py 375 &
python  ${path}/main.py 376 &
python  ${path}/main.py 377 &
python  ${path}/main.py 378 &
python  ${path}/main.py 379 &
python  ${path}/main.py 380 &
python  ${path}/main.py 381 &
python  ${path}/main.py 382 &
python  ${path}/main.py 383 &
python  ${path}/main.py 384 &
python  ${path}/main.py 385 &
python  ${path}/main.py 386 &
python  ${path}/main.py 387 &
python  ${path}/main.py 388 &
python  ${path}/main.py 389 &
python  ${path}/main.py 390 &
python  ${path}/main.py 391 &
python  ${path}/main.py 392 &
python  ${path}/main.py 393 &
python  ${path}/main.py 394 &
python  ${path}/main.py 395 &
wait
