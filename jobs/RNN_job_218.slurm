#!/bin/bash -l
#SBATCH -J RNN_218
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7812 &
python  ${path}/main.py 7813 &
python  ${path}/main.py 7814 &
python  ${path}/main.py 7815 &
python  ${path}/main.py 7816 &
python  ${path}/main.py 7817 &
python  ${path}/main.py 7818 &
python  ${path}/main.py 7819 &
python  ${path}/main.py 7820 &
python  ${path}/main.py 7821 &
python  ${path}/main.py 7822 &
python  ${path}/main.py 7823 &
python  ${path}/main.py 7824 &
python  ${path}/main.py 7825 &
python  ${path}/main.py 7826 &
python  ${path}/main.py 7827 &
python  ${path}/main.py 7828 &
python  ${path}/main.py 7829 &
python  ${path}/main.py 7830 &
python  ${path}/main.py 7831 &
python  ${path}/main.py 7832 &
python  ${path}/main.py 7833 &
python  ${path}/main.py 7834 &
python  ${path}/main.py 7835 &
python  ${path}/main.py 7836 &
python  ${path}/main.py 7837 &
python  ${path}/main.py 7838 &
python  ${path}/main.py 7839 &
python  ${path}/main.py 7840 &
python  ${path}/main.py 7841 &
python  ${path}/main.py 7842 &
python  ${path}/main.py 7843 &
python  ${path}/main.py 7844 &
python  ${path}/main.py 7845 &
python  ${path}/main.py 7846 &
python  ${path}/main.py 7847 &
wait
