#!/bin/bash -l
#SBATCH -J RNN_188
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6732 &
python  ${path}/main.py 6733 &
python  ${path}/main.py 6734 &
python  ${path}/main.py 6735 &
python  ${path}/main.py 6736 &
python  ${path}/main.py 6737 &
python  ${path}/main.py 6738 &
python  ${path}/main.py 6739 &
python  ${path}/main.py 6740 &
python  ${path}/main.py 6741 &
python  ${path}/main.py 6742 &
python  ${path}/main.py 6743 &
python  ${path}/main.py 6744 &
python  ${path}/main.py 6745 &
python  ${path}/main.py 6746 &
python  ${path}/main.py 6747 &
python  ${path}/main.py 6748 &
python  ${path}/main.py 6749 &
python  ${path}/main.py 6750 &
python  ${path}/main.py 6751 &
python  ${path}/main.py 6752 &
python  ${path}/main.py 6753 &
python  ${path}/main.py 6754 &
python  ${path}/main.py 6755 &
python  ${path}/main.py 6756 &
python  ${path}/main.py 6757 &
python  ${path}/main.py 6758 &
python  ${path}/main.py 6759 &
python  ${path}/main.py 6760 &
python  ${path}/main.py 6761 &
python  ${path}/main.py 6762 &
python  ${path}/main.py 6763 &
python  ${path}/main.py 6764 &
python  ${path}/main.py 6765 &
python  ${path}/main.py 6766 &
python  ${path}/main.py 6767 &
wait
