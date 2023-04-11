#!/bin/bash -l
#SBATCH -J RNN_213
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7632 &
python  ${path}/main.py 7633 &
python  ${path}/main.py 7634 &
python  ${path}/main.py 7635 &
python  ${path}/main.py 7636 &
python  ${path}/main.py 7637 &
python  ${path}/main.py 7638 &
python  ${path}/main.py 7639 &
python  ${path}/main.py 7640 &
python  ${path}/main.py 7641 &
python  ${path}/main.py 7642 &
python  ${path}/main.py 7643 &
python  ${path}/main.py 7644 &
python  ${path}/main.py 7645 &
python  ${path}/main.py 7646 &
python  ${path}/main.py 7647 &
python  ${path}/main.py 7648 &
python  ${path}/main.py 7649 &
python  ${path}/main.py 7650 &
python  ${path}/main.py 7651 &
python  ${path}/main.py 7652 &
python  ${path}/main.py 7653 &
python  ${path}/main.py 7654 &
python  ${path}/main.py 7655 &
python  ${path}/main.py 7656 &
python  ${path}/main.py 7657 &
python  ${path}/main.py 7658 &
python  ${path}/main.py 7659 &
python  ${path}/main.py 7660 &
python  ${path}/main.py 7661 &
python  ${path}/main.py 7662 &
python  ${path}/main.py 7663 &
python  ${path}/main.py 7664 &
python  ${path}/main.py 7665 &
python  ${path}/main.py 7666 &
python  ${path}/main.py 7667 &
wait
