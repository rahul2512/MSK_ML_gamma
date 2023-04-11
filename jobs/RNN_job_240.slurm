#!/bin/bash -l
#SBATCH -J RNN_240
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8604 &
python  ${path}/main.py 8605 &
python  ${path}/main.py 8606 &
python  ${path}/main.py 8607 &
python  ${path}/main.py 8608 &
python  ${path}/main.py 8609 &
python  ${path}/main.py 8610 &
python  ${path}/main.py 8611 &
python  ${path}/main.py 8612 &
python  ${path}/main.py 8613 &
python  ${path}/main.py 8614 &
python  ${path}/main.py 8615 &
python  ${path}/main.py 8616 &
python  ${path}/main.py 8617 &
python  ${path}/main.py 8618 &
python  ${path}/main.py 8619 &
python  ${path}/main.py 8620 &
python  ${path}/main.py 8621 &
python  ${path}/main.py 8622 &
python  ${path}/main.py 8623 &
python  ${path}/main.py 8624 &
python  ${path}/main.py 8625 &
python  ${path}/main.py 8626 &
python  ${path}/main.py 8627 &
python  ${path}/main.py 8628 &
python  ${path}/main.py 8629 &
python  ${path}/main.py 8630 &
python  ${path}/main.py 8631 &
python  ${path}/main.py 8632 &
python  ${path}/main.py 8633 &
python  ${path}/main.py 8634 &
python  ${path}/main.py 8635 &
python  ${path}/main.py 8636 &
python  ${path}/main.py 8637 &
python  ${path}/main.py 8638 &
python  ${path}/main.py 8639 &
wait
