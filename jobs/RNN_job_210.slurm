#!/bin/bash -l
#SBATCH -J RNN_210
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7524 &
python  ${path}/main.py 7525 &
python  ${path}/main.py 7526 &
python  ${path}/main.py 7527 &
python  ${path}/main.py 7528 &
python  ${path}/main.py 7529 &
python  ${path}/main.py 7530 &
python  ${path}/main.py 7531 &
python  ${path}/main.py 7532 &
python  ${path}/main.py 7533 &
python  ${path}/main.py 7534 &
python  ${path}/main.py 7535 &
python  ${path}/main.py 7536 &
python  ${path}/main.py 7537 &
python  ${path}/main.py 7538 &
python  ${path}/main.py 7539 &
python  ${path}/main.py 7540 &
python  ${path}/main.py 7541 &
python  ${path}/main.py 7542 &
python  ${path}/main.py 7543 &
python  ${path}/main.py 7544 &
python  ${path}/main.py 7545 &
python  ${path}/main.py 7546 &
python  ${path}/main.py 7547 &
python  ${path}/main.py 7548 &
python  ${path}/main.py 7549 &
python  ${path}/main.py 7550 &
python  ${path}/main.py 7551 &
python  ${path}/main.py 7552 &
python  ${path}/main.py 7553 &
python  ${path}/main.py 7554 &
python  ${path}/main.py 7555 &
python  ${path}/main.py 7556 &
python  ${path}/main.py 7557 &
python  ${path}/main.py 7558 &
python  ${path}/main.py 7559 &
wait
