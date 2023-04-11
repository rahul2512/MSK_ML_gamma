#!/bin/bash -l
#SBATCH -J RNN_38
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1332 &
python  ${path}/main.py 1333 &
python  ${path}/main.py 1334 &
python  ${path}/main.py 1335 &
python  ${path}/main.py 1336 &
python  ${path}/main.py 1337 &
python  ${path}/main.py 1338 &
python  ${path}/main.py 1339 &
python  ${path}/main.py 1340 &
python  ${path}/main.py 1341 &
python  ${path}/main.py 1342 &
python  ${path}/main.py 1343 &
python  ${path}/main.py 1344 &
python  ${path}/main.py 1345 &
python  ${path}/main.py 1346 &
python  ${path}/main.py 1347 &
python  ${path}/main.py 1348 &
python  ${path}/main.py 1349 &
python  ${path}/main.py 1350 &
python  ${path}/main.py 1351 &
python  ${path}/main.py 1352 &
python  ${path}/main.py 1353 &
python  ${path}/main.py 1354 &
python  ${path}/main.py 1355 &
python  ${path}/main.py 1356 &
python  ${path}/main.py 1357 &
python  ${path}/main.py 1358 &
python  ${path}/main.py 1359 &
python  ${path}/main.py 1360 &
python  ${path}/main.py 1361 &
python  ${path}/main.py 1362 &
python  ${path}/main.py 1363 &
python  ${path}/main.py 1364 &
python  ${path}/main.py 1365 &
python  ${path}/main.py 1366 &
python  ${path}/main.py 1367 &
wait
