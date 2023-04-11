#!/bin/bash -l
#SBATCH -J RNN_216
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7740 &
python  ${path}/main.py 7741 &
python  ${path}/main.py 7742 &
python  ${path}/main.py 7743 &
python  ${path}/main.py 7744 &
python  ${path}/main.py 7745 &
python  ${path}/main.py 7746 &
python  ${path}/main.py 7747 &
python  ${path}/main.py 7748 &
python  ${path}/main.py 7749 &
python  ${path}/main.py 7750 &
python  ${path}/main.py 7751 &
python  ${path}/main.py 7752 &
python  ${path}/main.py 7753 &
python  ${path}/main.py 7754 &
python  ${path}/main.py 7755 &
python  ${path}/main.py 7756 &
python  ${path}/main.py 7757 &
python  ${path}/main.py 7758 &
python  ${path}/main.py 7759 &
python  ${path}/main.py 7760 &
python  ${path}/main.py 7761 &
python  ${path}/main.py 7762 &
python  ${path}/main.py 7763 &
python  ${path}/main.py 7764 &
python  ${path}/main.py 7765 &
python  ${path}/main.py 7766 &
python  ${path}/main.py 7767 &
python  ${path}/main.py 7768 &
python  ${path}/main.py 7769 &
python  ${path}/main.py 7770 &
python  ${path}/main.py 7771 &
python  ${path}/main.py 7772 &
python  ${path}/main.py 7773 &
python  ${path}/main.py 7774 &
python  ${path}/main.py 7775 &
wait
