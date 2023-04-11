#!/bin/bash -l
#SBATCH -J RNN_214
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7668 &
python  ${path}/main.py 7669 &
python  ${path}/main.py 7670 &
python  ${path}/main.py 7671 &
python  ${path}/main.py 7672 &
python  ${path}/main.py 7673 &
python  ${path}/main.py 7674 &
python  ${path}/main.py 7675 &
python  ${path}/main.py 7676 &
python  ${path}/main.py 7677 &
python  ${path}/main.py 7678 &
python  ${path}/main.py 7679 &
python  ${path}/main.py 7680 &
python  ${path}/main.py 7681 &
python  ${path}/main.py 7682 &
python  ${path}/main.py 7683 &
python  ${path}/main.py 7684 &
python  ${path}/main.py 7685 &
python  ${path}/main.py 7686 &
python  ${path}/main.py 7687 &
python  ${path}/main.py 7688 &
python  ${path}/main.py 7689 &
python  ${path}/main.py 7690 &
python  ${path}/main.py 7691 &
python  ${path}/main.py 7692 &
python  ${path}/main.py 7693 &
python  ${path}/main.py 7694 &
python  ${path}/main.py 7695 &
python  ${path}/main.py 7696 &
python  ${path}/main.py 7697 &
python  ${path}/main.py 7698 &
python  ${path}/main.py 7699 &
python  ${path}/main.py 7700 &
python  ${path}/main.py 7701 &
python  ${path}/main.py 7702 &
python  ${path}/main.py 7703 &
wait
