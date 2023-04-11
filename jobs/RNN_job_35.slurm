#!/bin/bash -l
#SBATCH -J RNN_35
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1224 &
python  ${path}/main.py 1225 &
python  ${path}/main.py 1226 &
python  ${path}/main.py 1227 &
python  ${path}/main.py 1228 &
python  ${path}/main.py 1229 &
python  ${path}/main.py 1230 &
python  ${path}/main.py 1231 &
python  ${path}/main.py 1232 &
python  ${path}/main.py 1233 &
python  ${path}/main.py 1234 &
python  ${path}/main.py 1235 &
python  ${path}/main.py 1236 &
python  ${path}/main.py 1237 &
python  ${path}/main.py 1238 &
python  ${path}/main.py 1239 &
python  ${path}/main.py 1240 &
python  ${path}/main.py 1241 &
python  ${path}/main.py 1242 &
python  ${path}/main.py 1243 &
python  ${path}/main.py 1244 &
python  ${path}/main.py 1245 &
python  ${path}/main.py 1246 &
python  ${path}/main.py 1247 &
python  ${path}/main.py 1248 &
python  ${path}/main.py 1249 &
python  ${path}/main.py 1250 &
python  ${path}/main.py 1251 &
python  ${path}/main.py 1252 &
python  ${path}/main.py 1253 &
python  ${path}/main.py 1254 &
python  ${path}/main.py 1255 &
python  ${path}/main.py 1256 &
python  ${path}/main.py 1257 &
python  ${path}/main.py 1258 &
python  ${path}/main.py 1259 &
wait
