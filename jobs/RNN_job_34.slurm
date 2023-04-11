#!/bin/bash -l
#SBATCH -J RNN_34
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1188 &
python  ${path}/main.py 1189 &
python  ${path}/main.py 1190 &
python  ${path}/main.py 1191 &
python  ${path}/main.py 1192 &
python  ${path}/main.py 1193 &
python  ${path}/main.py 1194 &
python  ${path}/main.py 1195 &
python  ${path}/main.py 1196 &
python  ${path}/main.py 1197 &
python  ${path}/main.py 1198 &
python  ${path}/main.py 1199 &
python  ${path}/main.py 1200 &
python  ${path}/main.py 1201 &
python  ${path}/main.py 1202 &
python  ${path}/main.py 1203 &
python  ${path}/main.py 1204 &
python  ${path}/main.py 1205 &
python  ${path}/main.py 1206 &
python  ${path}/main.py 1207 &
python  ${path}/main.py 1208 &
python  ${path}/main.py 1209 &
python  ${path}/main.py 1210 &
python  ${path}/main.py 1211 &
python  ${path}/main.py 1212 &
python  ${path}/main.py 1213 &
python  ${path}/main.py 1214 &
python  ${path}/main.py 1215 &
python  ${path}/main.py 1216 &
python  ${path}/main.py 1217 &
python  ${path}/main.py 1218 &
python  ${path}/main.py 1219 &
python  ${path}/main.py 1220 &
python  ${path}/main.py 1221 &
python  ${path}/main.py 1222 &
python  ${path}/main.py 1223 &
wait
