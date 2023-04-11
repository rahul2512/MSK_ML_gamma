#!/bin/bash -l
#SBATCH -J RNN_135
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4824 &
python  ${path}/main.py 4825 &
python  ${path}/main.py 4826 &
python  ${path}/main.py 4827 &
python  ${path}/main.py 4828 &
python  ${path}/main.py 4829 &
python  ${path}/main.py 4830 &
python  ${path}/main.py 4831 &
python  ${path}/main.py 4832 &
python  ${path}/main.py 4833 &
python  ${path}/main.py 4834 &
python  ${path}/main.py 4835 &
python  ${path}/main.py 4836 &
python  ${path}/main.py 4837 &
python  ${path}/main.py 4838 &
python  ${path}/main.py 4839 &
python  ${path}/main.py 4840 &
python  ${path}/main.py 4841 &
python  ${path}/main.py 4842 &
python  ${path}/main.py 4843 &
python  ${path}/main.py 4844 &
python  ${path}/main.py 4845 &
python  ${path}/main.py 4846 &
python  ${path}/main.py 4847 &
python  ${path}/main.py 4848 &
python  ${path}/main.py 4849 &
python  ${path}/main.py 4850 &
python  ${path}/main.py 4851 &
python  ${path}/main.py 4852 &
python  ${path}/main.py 4853 &
python  ${path}/main.py 4854 &
python  ${path}/main.py 4855 &
python  ${path}/main.py 4856 &
python  ${path}/main.py 4857 &
python  ${path}/main.py 4858 &
python  ${path}/main.py 4859 &
wait
