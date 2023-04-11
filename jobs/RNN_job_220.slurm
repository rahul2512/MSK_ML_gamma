#!/bin/bash -l
#SBATCH -J RNN_220
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7884 &
python  ${path}/main.py 7885 &
python  ${path}/main.py 7886 &
python  ${path}/main.py 7887 &
python  ${path}/main.py 7888 &
python  ${path}/main.py 7889 &
python  ${path}/main.py 7890 &
python  ${path}/main.py 7891 &
python  ${path}/main.py 7892 &
python  ${path}/main.py 7893 &
python  ${path}/main.py 7894 &
python  ${path}/main.py 7895 &
python  ${path}/main.py 7896 &
python  ${path}/main.py 7897 &
python  ${path}/main.py 7898 &
python  ${path}/main.py 7899 &
python  ${path}/main.py 7900 &
python  ${path}/main.py 7901 &
python  ${path}/main.py 7902 &
python  ${path}/main.py 7903 &
python  ${path}/main.py 7904 &
python  ${path}/main.py 7905 &
python  ${path}/main.py 7906 &
python  ${path}/main.py 7907 &
python  ${path}/main.py 7908 &
python  ${path}/main.py 7909 &
python  ${path}/main.py 7910 &
python  ${path}/main.py 7911 &
python  ${path}/main.py 7912 &
python  ${path}/main.py 7913 &
python  ${path}/main.py 7914 &
python  ${path}/main.py 7915 &
python  ${path}/main.py 7916 &
python  ${path}/main.py 7917 &
python  ${path}/main.py 7918 &
python  ${path}/main.py 7919 &
wait
