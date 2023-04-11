#!/bin/bash -l
#SBATCH -J RNN_215
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7704 &
python  ${path}/main.py 7705 &
python  ${path}/main.py 7706 &
python  ${path}/main.py 7707 &
python  ${path}/main.py 7708 &
python  ${path}/main.py 7709 &
python  ${path}/main.py 7710 &
python  ${path}/main.py 7711 &
python  ${path}/main.py 7712 &
python  ${path}/main.py 7713 &
python  ${path}/main.py 7714 &
python  ${path}/main.py 7715 &
python  ${path}/main.py 7716 &
python  ${path}/main.py 7717 &
python  ${path}/main.py 7718 &
python  ${path}/main.py 7719 &
python  ${path}/main.py 7720 &
python  ${path}/main.py 7721 &
python  ${path}/main.py 7722 &
python  ${path}/main.py 7723 &
python  ${path}/main.py 7724 &
python  ${path}/main.py 7725 &
python  ${path}/main.py 7726 &
python  ${path}/main.py 7727 &
python  ${path}/main.py 7728 &
python  ${path}/main.py 7729 &
python  ${path}/main.py 7730 &
python  ${path}/main.py 7731 &
python  ${path}/main.py 7732 &
python  ${path}/main.py 7733 &
python  ${path}/main.py 7734 &
python  ${path}/main.py 7735 &
python  ${path}/main.py 7736 &
python  ${path}/main.py 7737 &
python  ${path}/main.py 7738 &
python  ${path}/main.py 7739 &
wait
