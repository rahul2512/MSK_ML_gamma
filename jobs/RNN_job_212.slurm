#!/bin/bash -l
#SBATCH -J RNN_212
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7596 &
python  ${path}/main.py 7597 &
python  ${path}/main.py 7598 &
python  ${path}/main.py 7599 &
python  ${path}/main.py 7600 &
python  ${path}/main.py 7601 &
python  ${path}/main.py 7602 &
python  ${path}/main.py 7603 &
python  ${path}/main.py 7604 &
python  ${path}/main.py 7605 &
python  ${path}/main.py 7606 &
python  ${path}/main.py 7607 &
python  ${path}/main.py 7608 &
python  ${path}/main.py 7609 &
python  ${path}/main.py 7610 &
python  ${path}/main.py 7611 &
python  ${path}/main.py 7612 &
python  ${path}/main.py 7613 &
python  ${path}/main.py 7614 &
python  ${path}/main.py 7615 &
python  ${path}/main.py 7616 &
python  ${path}/main.py 7617 &
python  ${path}/main.py 7618 &
python  ${path}/main.py 7619 &
python  ${path}/main.py 7620 &
python  ${path}/main.py 7621 &
python  ${path}/main.py 7622 &
python  ${path}/main.py 7623 &
python  ${path}/main.py 7624 &
python  ${path}/main.py 7625 &
python  ${path}/main.py 7626 &
python  ${path}/main.py 7627 &
python  ${path}/main.py 7628 &
python  ${path}/main.py 7629 &
python  ${path}/main.py 7630 &
python  ${path}/main.py 7631 &
wait
