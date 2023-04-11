#!/bin/bash -l
#SBATCH -J RNN_243
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8712 &
python  ${path}/main.py 8713 &
python  ${path}/main.py 8714 &
python  ${path}/main.py 8715 &
python  ${path}/main.py 8716 &
python  ${path}/main.py 8717 &
python  ${path}/main.py 8718 &
python  ${path}/main.py 8719 &
python  ${path}/main.py 8720 &
python  ${path}/main.py 8721 &
python  ${path}/main.py 8722 &
python  ${path}/main.py 8723 &
python  ${path}/main.py 8724 &
python  ${path}/main.py 8725 &
python  ${path}/main.py 8726 &
python  ${path}/main.py 8727 &
python  ${path}/main.py 8728 &
python  ${path}/main.py 8729 &
python  ${path}/main.py 8730 &
python  ${path}/main.py 8731 &
python  ${path}/main.py 8732 &
python  ${path}/main.py 8733 &
python  ${path}/main.py 8734 &
python  ${path}/main.py 8735 &
python  ${path}/main.py 8736 &
python  ${path}/main.py 8737 &
python  ${path}/main.py 8738 &
python  ${path}/main.py 8739 &
python  ${path}/main.py 8740 &
python  ${path}/main.py 8741 &
python  ${path}/main.py 8742 &
python  ${path}/main.py 8743 &
python  ${path}/main.py 8744 &
python  ${path}/main.py 8745 &
python  ${path}/main.py 8746 &
python  ${path}/main.py 8747 &
wait
