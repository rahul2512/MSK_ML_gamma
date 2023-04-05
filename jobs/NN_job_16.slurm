#!/bin/bash -l
#SBATCH -J NN_16
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
python  ${path}/main.py 1116 &
python  ${path}/main.py 1117 &
python  ${path}/main.py 1118 &
python  ${path}/main.py 1119 &
python  ${path}/main.py 1120 &
python  ${path}/main.py 1121 &
python  ${path}/main.py 1122 &
python  ${path}/main.py 1123 &
python  ${path}/main.py 1124 &
python  ${path}/main.py 1125 &
python  ${path}/main.py 1126 &
python  ${path}/main.py 1127 &
python  ${path}/main.py 1128 &
python  ${path}/main.py 1129 &
python  ${path}/main.py 1130 &
python  ${path}/main.py 1131 &
python  ${path}/main.py 1132 &
python  ${path}/main.py 1133 &
python  ${path}/main.py 1134 &
python  ${path}/main.py 1135 &
python  ${path}/main.py 1136 &
python  ${path}/main.py 1137 &
python  ${path}/main.py 1138 &
python  ${path}/main.py 1139 &
python  ${path}/main.py 1140 &
python  ${path}/main.py 1141 &
python  ${path}/main.py 1142 &
python  ${path}/main.py 1143 &
python  ${path}/main.py 1144 &
python  ${path}/main.py 1145 &
python  ${path}/main.py 1146 &
python  ${path}/main.py 1147 &
python  ${path}/main.py 1148 &
python  ${path}/main.py 1149 &
python  ${path}/main.py 1150 &
python  ${path}/main.py 1151 &
wait
