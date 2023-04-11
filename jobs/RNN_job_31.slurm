#!/bin/bash -l
#SBATCH -J RNN_31
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1080 &
python  ${path}/main.py 1081 &
python  ${path}/main.py 1082 &
python  ${path}/main.py 1083 &
python  ${path}/main.py 1084 &
python  ${path}/main.py 1085 &
python  ${path}/main.py 1086 &
python  ${path}/main.py 1087 &
python  ${path}/main.py 1088 &
python  ${path}/main.py 1089 &
python  ${path}/main.py 1090 &
python  ${path}/main.py 1091 &
python  ${path}/main.py 1092 &
python  ${path}/main.py 1093 &
python  ${path}/main.py 1094 &
python  ${path}/main.py 1095 &
python  ${path}/main.py 1096 &
python  ${path}/main.py 1097 &
python  ${path}/main.py 1098 &
python  ${path}/main.py 1099 &
python  ${path}/main.py 1100 &
python  ${path}/main.py 1101 &
python  ${path}/main.py 1102 &
python  ${path}/main.py 1103 &
python  ${path}/main.py 1104 &
python  ${path}/main.py 1105 &
python  ${path}/main.py 1106 &
python  ${path}/main.py 1107 &
python  ${path}/main.py 1108 &
python  ${path}/main.py 1109 &
python  ${path}/main.py 1110 &
python  ${path}/main.py 1111 &
python  ${path}/main.py 1112 &
python  ${path}/main.py 1113 &
python  ${path}/main.py 1114 &
python  ${path}/main.py 1115 &
wait
