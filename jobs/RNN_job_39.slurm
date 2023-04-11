#!/bin/bash -l
#SBATCH -J RNN_39
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1368 &
python  ${path}/main.py 1369 &
python  ${path}/main.py 1370 &
python  ${path}/main.py 1371 &
python  ${path}/main.py 1372 &
python  ${path}/main.py 1373 &
python  ${path}/main.py 1374 &
python  ${path}/main.py 1375 &
python  ${path}/main.py 1376 &
python  ${path}/main.py 1377 &
python  ${path}/main.py 1378 &
python  ${path}/main.py 1379 &
python  ${path}/main.py 1380 &
python  ${path}/main.py 1381 &
python  ${path}/main.py 1382 &
python  ${path}/main.py 1383 &
python  ${path}/main.py 1384 &
python  ${path}/main.py 1385 &
python  ${path}/main.py 1386 &
python  ${path}/main.py 1387 &
python  ${path}/main.py 1388 &
python  ${path}/main.py 1389 &
python  ${path}/main.py 1390 &
python  ${path}/main.py 1391 &
python  ${path}/main.py 1392 &
python  ${path}/main.py 1393 &
python  ${path}/main.py 1394 &
python  ${path}/main.py 1395 &
python  ${path}/main.py 1396 &
python  ${path}/main.py 1397 &
python  ${path}/main.py 1398 &
python  ${path}/main.py 1399 &
python  ${path}/main.py 1400 &
python  ${path}/main.py 1401 &
python  ${path}/main.py 1402 &
python  ${path}/main.py 1403 &
wait
