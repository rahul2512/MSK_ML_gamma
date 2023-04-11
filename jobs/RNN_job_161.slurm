#!/bin/bash -l
#SBATCH -J RNN_161
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5760 &
python  ${path}/main.py 5761 &
python  ${path}/main.py 5762 &
python  ${path}/main.py 5763 &
python  ${path}/main.py 5764 &
python  ${path}/main.py 5765 &
python  ${path}/main.py 5766 &
python  ${path}/main.py 5767 &
python  ${path}/main.py 5768 &
python  ${path}/main.py 5769 &
python  ${path}/main.py 5770 &
python  ${path}/main.py 5771 &
python  ${path}/main.py 5772 &
python  ${path}/main.py 5773 &
python  ${path}/main.py 5774 &
python  ${path}/main.py 5775 &
python  ${path}/main.py 5776 &
python  ${path}/main.py 5777 &
python  ${path}/main.py 5778 &
python  ${path}/main.py 5779 &
python  ${path}/main.py 5780 &
python  ${path}/main.py 5781 &
python  ${path}/main.py 5782 &
python  ${path}/main.py 5783 &
python  ${path}/main.py 5784 &
python  ${path}/main.py 5785 &
python  ${path}/main.py 5786 &
python  ${path}/main.py 5787 &
python  ${path}/main.py 5788 &
python  ${path}/main.py 5789 &
python  ${path}/main.py 5790 &
python  ${path}/main.py 5791 &
python  ${path}/main.py 5792 &
python  ${path}/main.py 5793 &
python  ${path}/main.py 5794 &
python  ${path}/main.py 5795 &
wait
