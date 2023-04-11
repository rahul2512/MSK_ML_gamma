#!/bin/bash -l
#SBATCH -J RNN_164
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5868 &
python  ${path}/main.py 5869 &
python  ${path}/main.py 5870 &
python  ${path}/main.py 5871 &
python  ${path}/main.py 5872 &
python  ${path}/main.py 5873 &
python  ${path}/main.py 5874 &
python  ${path}/main.py 5875 &
python  ${path}/main.py 5876 &
python  ${path}/main.py 5877 &
python  ${path}/main.py 5878 &
python  ${path}/main.py 5879 &
python  ${path}/main.py 5880 &
python  ${path}/main.py 5881 &
python  ${path}/main.py 5882 &
python  ${path}/main.py 5883 &
python  ${path}/main.py 5884 &
python  ${path}/main.py 5885 &
python  ${path}/main.py 5886 &
python  ${path}/main.py 5887 &
python  ${path}/main.py 5888 &
python  ${path}/main.py 5889 &
python  ${path}/main.py 5890 &
python  ${path}/main.py 5891 &
python  ${path}/main.py 5892 &
python  ${path}/main.py 5893 &
python  ${path}/main.py 5894 &
python  ${path}/main.py 5895 &
python  ${path}/main.py 5896 &
python  ${path}/main.py 5897 &
python  ${path}/main.py 5898 &
python  ${path}/main.py 5899 &
python  ${path}/main.py 5900 &
python  ${path}/main.py 5901 &
python  ${path}/main.py 5902 &
python  ${path}/main.py 5903 &
wait
