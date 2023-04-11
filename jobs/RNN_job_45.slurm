#!/bin/bash -l
#SBATCH -J RNN_45
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1584 &
python  ${path}/main.py 1585 &
python  ${path}/main.py 1586 &
python  ${path}/main.py 1587 &
python  ${path}/main.py 1588 &
python  ${path}/main.py 1589 &
python  ${path}/main.py 1590 &
python  ${path}/main.py 1591 &
python  ${path}/main.py 1592 &
python  ${path}/main.py 1593 &
python  ${path}/main.py 1594 &
python  ${path}/main.py 1595 &
python  ${path}/main.py 1596 &
python  ${path}/main.py 1597 &
python  ${path}/main.py 1598 &
python  ${path}/main.py 1599 &
python  ${path}/main.py 1600 &
python  ${path}/main.py 1601 &
python  ${path}/main.py 1602 &
python  ${path}/main.py 1603 &
python  ${path}/main.py 1604 &
python  ${path}/main.py 1605 &
python  ${path}/main.py 1606 &
python  ${path}/main.py 1607 &
python  ${path}/main.py 1608 &
python  ${path}/main.py 1609 &
python  ${path}/main.py 1610 &
python  ${path}/main.py 1611 &
python  ${path}/main.py 1612 &
python  ${path}/main.py 1613 &
python  ${path}/main.py 1614 &
python  ${path}/main.py 1615 &
python  ${path}/main.py 1616 &
python  ${path}/main.py 1617 &
python  ${path}/main.py 1618 &
python  ${path}/main.py 1619 &
wait
