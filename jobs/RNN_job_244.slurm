#!/bin/bash -l
#SBATCH -J RNN_244
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8748 &
python  ${path}/main.py 8749 &
python  ${path}/main.py 8750 &
python  ${path}/main.py 8751 &
python  ${path}/main.py 8752 &
python  ${path}/main.py 8753 &
python  ${path}/main.py 8754 &
python  ${path}/main.py 8755 &
python  ${path}/main.py 8756 &
python  ${path}/main.py 8757 &
python  ${path}/main.py 8758 &
python  ${path}/main.py 8759 &
python  ${path}/main.py 8760 &
python  ${path}/main.py 8761 &
python  ${path}/main.py 8762 &
python  ${path}/main.py 8763 &
python  ${path}/main.py 8764 &
python  ${path}/main.py 8765 &
python  ${path}/main.py 8766 &
python  ${path}/main.py 8767 &
python  ${path}/main.py 8768 &
python  ${path}/main.py 8769 &
python  ${path}/main.py 8770 &
python  ${path}/main.py 8771 &
python  ${path}/main.py 8772 &
python  ${path}/main.py 8773 &
python  ${path}/main.py 8774 &
python  ${path}/main.py 8775 &
python  ${path}/main.py 8776 &
python  ${path}/main.py 8777 &
python  ${path}/main.py 8778 &
python  ${path}/main.py 8779 &
python  ${path}/main.py 8780 &
python  ${path}/main.py 8781 &
python  ${path}/main.py 8782 &
python  ${path}/main.py 8783 &
wait
