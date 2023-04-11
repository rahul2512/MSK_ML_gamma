#!/bin/bash -l
#SBATCH -J RNN_30
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1044 &
python  ${path}/main.py 1045 &
python  ${path}/main.py 1046 &
python  ${path}/main.py 1047 &
python  ${path}/main.py 1048 &
python  ${path}/main.py 1049 &
python  ${path}/main.py 1050 &
python  ${path}/main.py 1051 &
python  ${path}/main.py 1052 &
python  ${path}/main.py 1053 &
python  ${path}/main.py 1054 &
python  ${path}/main.py 1055 &
python  ${path}/main.py 1056 &
python  ${path}/main.py 1057 &
python  ${path}/main.py 1058 &
python  ${path}/main.py 1059 &
python  ${path}/main.py 1060 &
python  ${path}/main.py 1061 &
python  ${path}/main.py 1062 &
python  ${path}/main.py 1063 &
python  ${path}/main.py 1064 &
python  ${path}/main.py 1065 &
python  ${path}/main.py 1066 &
python  ${path}/main.py 1067 &
python  ${path}/main.py 1068 &
python  ${path}/main.py 1069 &
python  ${path}/main.py 1070 &
python  ${path}/main.py 1071 &
python  ${path}/main.py 1072 &
python  ${path}/main.py 1073 &
python  ${path}/main.py 1074 &
python  ${path}/main.py 1075 &
python  ${path}/main.py 1076 &
python  ${path}/main.py 1077 &
python  ${path}/main.py 1078 &
python  ${path}/main.py 1079 &
wait
