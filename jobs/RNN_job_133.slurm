#!/bin/bash -l
#SBATCH -J RNN_133
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4752 &
python  ${path}/main.py 4753 &
python  ${path}/main.py 4754 &
python  ${path}/main.py 4755 &
python  ${path}/main.py 4756 &
python  ${path}/main.py 4757 &
python  ${path}/main.py 4758 &
python  ${path}/main.py 4759 &
python  ${path}/main.py 4760 &
python  ${path}/main.py 4761 &
python  ${path}/main.py 4762 &
python  ${path}/main.py 4763 &
python  ${path}/main.py 4764 &
python  ${path}/main.py 4765 &
python  ${path}/main.py 4766 &
python  ${path}/main.py 4767 &
python  ${path}/main.py 4768 &
python  ${path}/main.py 4769 &
python  ${path}/main.py 4770 &
python  ${path}/main.py 4771 &
python  ${path}/main.py 4772 &
python  ${path}/main.py 4773 &
python  ${path}/main.py 4774 &
python  ${path}/main.py 4775 &
python  ${path}/main.py 4776 &
python  ${path}/main.py 4777 &
python  ${path}/main.py 4778 &
python  ${path}/main.py 4779 &
python  ${path}/main.py 4780 &
python  ${path}/main.py 4781 &
python  ${path}/main.py 4782 &
python  ${path}/main.py 4783 &
python  ${path}/main.py 4784 &
python  ${path}/main.py 4785 &
python  ${path}/main.py 4786 &
python  ${path}/main.py 4787 &
wait
