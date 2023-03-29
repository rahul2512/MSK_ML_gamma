#!/bin/bash -l
#SBATCH -J CNN_31
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 720 &
python  ${path}/main.py 721 &
python  ${path}/main.py 722 &
python  ${path}/main.py 723 &
python  ${path}/main.py 724 &
python  ${path}/main.py 725 &
python  ${path}/main.py 726 &
python  ${path}/main.py 727 &
python  ${path}/main.py 728 &
python  ${path}/main.py 729 &
python  ${path}/main.py 730 &
python  ${path}/main.py 731 &
python  ${path}/main.py 732 &
python  ${path}/main.py 733 &
python  ${path}/main.py 734 &
python  ${path}/main.py 735 &
python  ${path}/main.py 736 &
python  ${path}/main.py 737 &
python  ${path}/main.py 738 &
python  ${path}/main.py 739 &
python  ${path}/main.py 740 &
python  ${path}/main.py 741 &
python  ${path}/main.py 742 &
python  ${path}/main.py 743 &
wait
