#!/bin/bash -l
#SBATCH -J RNN_247
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8856 &
python  ${path}/main.py 8857 &
python  ${path}/main.py 8858 &
python  ${path}/main.py 8859 &
python  ${path}/main.py 8860 &
python  ${path}/main.py 8861 &
python  ${path}/main.py 8862 &
python  ${path}/main.py 8863 &
python  ${path}/main.py 8864 &
python  ${path}/main.py 8865 &
python  ${path}/main.py 8866 &
python  ${path}/main.py 8867 &
python  ${path}/main.py 8868 &
python  ${path}/main.py 8869 &
python  ${path}/main.py 8870 &
python  ${path}/main.py 8871 &
python  ${path}/main.py 8872 &
python  ${path}/main.py 8873 &
python  ${path}/main.py 8874 &
python  ${path}/main.py 8875 &
python  ${path}/main.py 8876 &
python  ${path}/main.py 8877 &
python  ${path}/main.py 8878 &
python  ${path}/main.py 8879 &
python  ${path}/main.py 8880 &
python  ${path}/main.py 8881 &
python  ${path}/main.py 8882 &
python  ${path}/main.py 8883 &
python  ${path}/main.py 8884 &
python  ${path}/main.py 8885 &
python  ${path}/main.py 8886 &
python  ${path}/main.py 8887 &
python  ${path}/main.py 8888 &
python  ${path}/main.py 8889 &
python  ${path}/main.py 8890 &
python  ${path}/main.py 8891 &
wait
