#!/bin/bash -l
#SBATCH -J RNN_40
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1404 &
python  ${path}/main.py 1405 &
python  ${path}/main.py 1406 &
python  ${path}/main.py 1407 &
python  ${path}/main.py 1408 &
python  ${path}/main.py 1409 &
python  ${path}/main.py 1410 &
python  ${path}/main.py 1411 &
python  ${path}/main.py 1412 &
python  ${path}/main.py 1413 &
python  ${path}/main.py 1414 &
python  ${path}/main.py 1415 &
python  ${path}/main.py 1416 &
python  ${path}/main.py 1417 &
python  ${path}/main.py 1418 &
python  ${path}/main.py 1419 &
python  ${path}/main.py 1420 &
python  ${path}/main.py 1421 &
python  ${path}/main.py 1422 &
python  ${path}/main.py 1423 &
python  ${path}/main.py 1424 &
python  ${path}/main.py 1425 &
python  ${path}/main.py 1426 &
python  ${path}/main.py 1427 &
python  ${path}/main.py 1428 &
python  ${path}/main.py 1429 &
python  ${path}/main.py 1430 &
python  ${path}/main.py 1431 &
python  ${path}/main.py 1432 &
python  ${path}/main.py 1433 &
python  ${path}/main.py 1434 &
python  ${path}/main.py 1435 &
python  ${path}/main.py 1436 &
python  ${path}/main.py 1437 &
python  ${path}/main.py 1438 &
python  ${path}/main.py 1439 &
wait
