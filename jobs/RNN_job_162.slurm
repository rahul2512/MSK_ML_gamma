#!/bin/bash -l
#SBATCH -J RNN_162
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5796 &
python  ${path}/main.py 5797 &
python  ${path}/main.py 5798 &
python  ${path}/main.py 5799 &
python  ${path}/main.py 5800 &
python  ${path}/main.py 5801 &
python  ${path}/main.py 5802 &
python  ${path}/main.py 5803 &
python  ${path}/main.py 5804 &
python  ${path}/main.py 5805 &
python  ${path}/main.py 5806 &
python  ${path}/main.py 5807 &
python  ${path}/main.py 5808 &
python  ${path}/main.py 5809 &
python  ${path}/main.py 5810 &
python  ${path}/main.py 5811 &
python  ${path}/main.py 5812 &
python  ${path}/main.py 5813 &
python  ${path}/main.py 5814 &
python  ${path}/main.py 5815 &
python  ${path}/main.py 5816 &
python  ${path}/main.py 5817 &
python  ${path}/main.py 5818 &
python  ${path}/main.py 5819 &
python  ${path}/main.py 5820 &
python  ${path}/main.py 5821 &
python  ${path}/main.py 5822 &
python  ${path}/main.py 5823 &
python  ${path}/main.py 5824 &
python  ${path}/main.py 5825 &
python  ${path}/main.py 5826 &
python  ${path}/main.py 5827 &
python  ${path}/main.py 5828 &
python  ${path}/main.py 5829 &
python  ${path}/main.py 5830 &
python  ${path}/main.py 5831 &
wait
