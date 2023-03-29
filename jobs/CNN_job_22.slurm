#!/bin/bash -l
#SBATCH -J CNN_22
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 504 &
python  ${path}/main.py 505 &
python  ${path}/main.py 506 &
python  ${path}/main.py 507 &
python  ${path}/main.py 508 &
python  ${path}/main.py 509 &
python  ${path}/main.py 510 &
python  ${path}/main.py 511 &
python  ${path}/main.py 512 &
python  ${path}/main.py 513 &
python  ${path}/main.py 514 &
python  ${path}/main.py 515 &
python  ${path}/main.py 516 &
python  ${path}/main.py 517 &
python  ${path}/main.py 518 &
python  ${path}/main.py 519 &
python  ${path}/main.py 520 &
python  ${path}/main.py 521 &
python  ${path}/main.py 522 &
python  ${path}/main.py 523 &
python  ${path}/main.py 524 &
python  ${path}/main.py 525 &
python  ${path}/main.py 526 &
python  ${path}/main.py 527 &
wait
