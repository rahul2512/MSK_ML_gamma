#!/bin/bash -l
#SBATCH -J RNN_267
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9576 &
python  ${path}/main.py 9577 &
python  ${path}/main.py 9578 &
python  ${path}/main.py 9579 &
python  ${path}/main.py 9580 &
python  ${path}/main.py 9581 &
python  ${path}/main.py 9582 &
python  ${path}/main.py 9583 &
python  ${path}/main.py 9584 &
python  ${path}/main.py 9585 &
python  ${path}/main.py 9586 &
python  ${path}/main.py 9587 &
python  ${path}/main.py 9588 &
python  ${path}/main.py 9589 &
python  ${path}/main.py 9590 &
python  ${path}/main.py 9591 &
python  ${path}/main.py 9592 &
python  ${path}/main.py 9593 &
python  ${path}/main.py 9594 &
python  ${path}/main.py 9595 &
python  ${path}/main.py 9596 &
python  ${path}/main.py 9597 &
python  ${path}/main.py 9598 &
python  ${path}/main.py 9599 &
python  ${path}/main.py 9600 &
python  ${path}/main.py 9601 &
python  ${path}/main.py 9602 &
python  ${path}/main.py 9603 &
python  ${path}/main.py 9604 &
python  ${path}/main.py 9605 &
python  ${path}/main.py 9606 &
python  ${path}/main.py 9607 &
python  ${path}/main.py 9608 &
python  ${path}/main.py 9609 &
python  ${path}/main.py 9610 &
python  ${path}/main.py 9611 &
wait
