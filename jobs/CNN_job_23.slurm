#!/bin/bash -l
#SBATCH -J CNN_23
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 528 &
python  ${path}/main.py 529 &
python  ${path}/main.py 530 &
python  ${path}/main.py 531 &
python  ${path}/main.py 532 &
python  ${path}/main.py 533 &
python  ${path}/main.py 534 &
python  ${path}/main.py 535 &
python  ${path}/main.py 536 &
python  ${path}/main.py 537 &
python  ${path}/main.py 538 &
python  ${path}/main.py 539 &
python  ${path}/main.py 540 &
python  ${path}/main.py 541 &
python  ${path}/main.py 542 &
python  ${path}/main.py 543 &
python  ${path}/main.py 544 &
python  ${path}/main.py 545 &
python  ${path}/main.py 546 &
python  ${path}/main.py 547 &
python  ${path}/main.py 548 &
python  ${path}/main.py 549 &
python  ${path}/main.py 550 &
python  ${path}/main.py 551 &
wait
