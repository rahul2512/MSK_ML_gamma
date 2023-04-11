#!/bin/bash -l
#SBATCH -J RNN_268
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9612 &
python  ${path}/main.py 9613 &
python  ${path}/main.py 9614 &
python  ${path}/main.py 9615 &
python  ${path}/main.py 9616 &
python  ${path}/main.py 9617 &
python  ${path}/main.py 9618 &
python  ${path}/main.py 9619 &
python  ${path}/main.py 9620 &
python  ${path}/main.py 9621 &
python  ${path}/main.py 9622 &
python  ${path}/main.py 9623 &
python  ${path}/main.py 9624 &
python  ${path}/main.py 9625 &
python  ${path}/main.py 9626 &
python  ${path}/main.py 9627 &
python  ${path}/main.py 9628 &
python  ${path}/main.py 9629 &
python  ${path}/main.py 9630 &
python  ${path}/main.py 9631 &
python  ${path}/main.py 9632 &
python  ${path}/main.py 9633 &
python  ${path}/main.py 9634 &
python  ${path}/main.py 9635 &
python  ${path}/main.py 9636 &
python  ${path}/main.py 9637 &
python  ${path}/main.py 9638 &
python  ${path}/main.py 9639 &
python  ${path}/main.py 9640 &
python  ${path}/main.py 9641 &
python  ${path}/main.py 9642 &
python  ${path}/main.py 9643 &
python  ${path}/main.py 9644 &
python  ${path}/main.py 9645 &
python  ${path}/main.py 9646 &
python  ${path}/main.py 9647 &
wait
