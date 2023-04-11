#!/bin/bash -l
#SBATCH -J RNN_153
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5472 &
python  ${path}/main.py 5473 &
python  ${path}/main.py 5474 &
python  ${path}/main.py 5475 &
python  ${path}/main.py 5476 &
python  ${path}/main.py 5477 &
python  ${path}/main.py 5478 &
python  ${path}/main.py 5479 &
python  ${path}/main.py 5480 &
python  ${path}/main.py 5481 &
python  ${path}/main.py 5482 &
python  ${path}/main.py 5483 &
python  ${path}/main.py 5484 &
python  ${path}/main.py 5485 &
python  ${path}/main.py 5486 &
python  ${path}/main.py 5487 &
python  ${path}/main.py 5488 &
python  ${path}/main.py 5489 &
python  ${path}/main.py 5490 &
python  ${path}/main.py 5491 &
python  ${path}/main.py 5492 &
python  ${path}/main.py 5493 &
python  ${path}/main.py 5494 &
python  ${path}/main.py 5495 &
python  ${path}/main.py 5496 &
python  ${path}/main.py 5497 &
python  ${path}/main.py 5498 &
python  ${path}/main.py 5499 &
python  ${path}/main.py 5500 &
python  ${path}/main.py 5501 &
python  ${path}/main.py 5502 &
python  ${path}/main.py 5503 &
python  ${path}/main.py 5504 &
python  ${path}/main.py 5505 &
python  ${path}/main.py 5506 &
python  ${path}/main.py 5507 &
wait
