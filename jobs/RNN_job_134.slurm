#!/bin/bash -l
#SBATCH -J RNN_134
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4788 &
python  ${path}/main.py 4789 &
python  ${path}/main.py 4790 &
python  ${path}/main.py 4791 &
python  ${path}/main.py 4792 &
python  ${path}/main.py 4793 &
python  ${path}/main.py 4794 &
python  ${path}/main.py 4795 &
python  ${path}/main.py 4796 &
python  ${path}/main.py 4797 &
python  ${path}/main.py 4798 &
python  ${path}/main.py 4799 &
python  ${path}/main.py 4800 &
python  ${path}/main.py 4801 &
python  ${path}/main.py 4802 &
python  ${path}/main.py 4803 &
python  ${path}/main.py 4804 &
python  ${path}/main.py 4805 &
python  ${path}/main.py 4806 &
python  ${path}/main.py 4807 &
python  ${path}/main.py 4808 &
python  ${path}/main.py 4809 &
python  ${path}/main.py 4810 &
python  ${path}/main.py 4811 &
python  ${path}/main.py 4812 &
python  ${path}/main.py 4813 &
python  ${path}/main.py 4814 &
python  ${path}/main.py 4815 &
python  ${path}/main.py 4816 &
python  ${path}/main.py 4817 &
python  ${path}/main.py 4818 &
python  ${path}/main.py 4819 &
python  ${path}/main.py 4820 &
python  ${path}/main.py 4821 &
python  ${path}/main.py 4822 &
python  ${path}/main.py 4823 &
wait
