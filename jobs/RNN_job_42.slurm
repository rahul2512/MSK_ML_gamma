#!/bin/bash -l
#SBATCH -J RNN_42
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1476 &
python  ${path}/main.py 1477 &
python  ${path}/main.py 1478 &
python  ${path}/main.py 1479 &
python  ${path}/main.py 1480 &
python  ${path}/main.py 1481 &
python  ${path}/main.py 1482 &
python  ${path}/main.py 1483 &
python  ${path}/main.py 1484 &
python  ${path}/main.py 1485 &
python  ${path}/main.py 1486 &
python  ${path}/main.py 1487 &
python  ${path}/main.py 1488 &
python  ${path}/main.py 1489 &
python  ${path}/main.py 1490 &
python  ${path}/main.py 1491 &
python  ${path}/main.py 1492 &
python  ${path}/main.py 1493 &
python  ${path}/main.py 1494 &
python  ${path}/main.py 1495 &
python  ${path}/main.py 1496 &
python  ${path}/main.py 1497 &
python  ${path}/main.py 1498 &
python  ${path}/main.py 1499 &
python  ${path}/main.py 1500 &
python  ${path}/main.py 1501 &
python  ${path}/main.py 1502 &
python  ${path}/main.py 1503 &
python  ${path}/main.py 1504 &
python  ${path}/main.py 1505 &
python  ${path}/main.py 1506 &
python  ${path}/main.py 1507 &
python  ${path}/main.py 1508 &
python  ${path}/main.py 1509 &
python  ${path}/main.py 1510 &
python  ${path}/main.py 1511 &
wait
