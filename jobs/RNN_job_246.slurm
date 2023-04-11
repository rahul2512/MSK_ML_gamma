#!/bin/bash -l
#SBATCH -J RNN_246
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8820 &
python  ${path}/main.py 8821 &
python  ${path}/main.py 8822 &
python  ${path}/main.py 8823 &
python  ${path}/main.py 8824 &
python  ${path}/main.py 8825 &
python  ${path}/main.py 8826 &
python  ${path}/main.py 8827 &
python  ${path}/main.py 8828 &
python  ${path}/main.py 8829 &
python  ${path}/main.py 8830 &
python  ${path}/main.py 8831 &
python  ${path}/main.py 8832 &
python  ${path}/main.py 8833 &
python  ${path}/main.py 8834 &
python  ${path}/main.py 8835 &
python  ${path}/main.py 8836 &
python  ${path}/main.py 8837 &
python  ${path}/main.py 8838 &
python  ${path}/main.py 8839 &
python  ${path}/main.py 8840 &
python  ${path}/main.py 8841 &
python  ${path}/main.py 8842 &
python  ${path}/main.py 8843 &
python  ${path}/main.py 8844 &
python  ${path}/main.py 8845 &
python  ${path}/main.py 8846 &
python  ${path}/main.py 8847 &
python  ${path}/main.py 8848 &
python  ${path}/main.py 8849 &
python  ${path}/main.py 8850 &
python  ${path}/main.py 8851 &
python  ${path}/main.py 8852 &
python  ${path}/main.py 8853 &
python  ${path}/main.py 8854 &
python  ${path}/main.py 8855 &
wait
