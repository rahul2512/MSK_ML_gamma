#!/bin/bash -l
#SBATCH -J CNN_21
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 480 &
python  ${path}/main.py 481 &
python  ${path}/main.py 482 &
python  ${path}/main.py 483 &
python  ${path}/main.py 484 &
python  ${path}/main.py 485 &
python  ${path}/main.py 486 &
python  ${path}/main.py 487 &
python  ${path}/main.py 488 &
python  ${path}/main.py 489 &
python  ${path}/main.py 490 &
python  ${path}/main.py 491 &
python  ${path}/main.py 492 &
python  ${path}/main.py 493 &
python  ${path}/main.py 494 &
python  ${path}/main.py 495 &
python  ${path}/main.py 496 &
python  ${path}/main.py 497 &
python  ${path}/main.py 498 &
python  ${path}/main.py 499 &
python  ${path}/main.py 500 &
python  ${path}/main.py 501 &
python  ${path}/main.py 502 &
python  ${path}/main.py 503 &
wait
