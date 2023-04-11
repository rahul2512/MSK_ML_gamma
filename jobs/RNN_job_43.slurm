#!/bin/bash -l
#SBATCH -J RNN_43
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1512 &
python  ${path}/main.py 1513 &
python  ${path}/main.py 1514 &
python  ${path}/main.py 1515 &
python  ${path}/main.py 1516 &
python  ${path}/main.py 1517 &
python  ${path}/main.py 1518 &
python  ${path}/main.py 1519 &
python  ${path}/main.py 1520 &
python  ${path}/main.py 1521 &
python  ${path}/main.py 1522 &
python  ${path}/main.py 1523 &
python  ${path}/main.py 1524 &
python  ${path}/main.py 1525 &
python  ${path}/main.py 1526 &
python  ${path}/main.py 1527 &
python  ${path}/main.py 1528 &
python  ${path}/main.py 1529 &
python  ${path}/main.py 1530 &
python  ${path}/main.py 1531 &
python  ${path}/main.py 1532 &
python  ${path}/main.py 1533 &
python  ${path}/main.py 1534 &
python  ${path}/main.py 1535 &
python  ${path}/main.py 1536 &
python  ${path}/main.py 1537 &
python  ${path}/main.py 1538 &
python  ${path}/main.py 1539 &
python  ${path}/main.py 1540 &
python  ${path}/main.py 1541 &
python  ${path}/main.py 1542 &
python  ${path}/main.py 1543 &
python  ${path}/main.py 1544 &
python  ${path}/main.py 1545 &
python  ${path}/main.py 1546 &
python  ${path}/main.py 1547 &
wait
