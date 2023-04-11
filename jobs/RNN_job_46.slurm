#!/bin/bash -l
#SBATCH -J RNN_46
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1620 &
python  ${path}/main.py 1621 &
python  ${path}/main.py 1622 &
python  ${path}/main.py 1623 &
python  ${path}/main.py 1624 &
python  ${path}/main.py 1625 &
python  ${path}/main.py 1626 &
python  ${path}/main.py 1627 &
python  ${path}/main.py 1628 &
python  ${path}/main.py 1629 &
python  ${path}/main.py 1630 &
python  ${path}/main.py 1631 &
python  ${path}/main.py 1632 &
python  ${path}/main.py 1633 &
python  ${path}/main.py 1634 &
python  ${path}/main.py 1635 &
python  ${path}/main.py 1636 &
python  ${path}/main.py 1637 &
python  ${path}/main.py 1638 &
python  ${path}/main.py 1639 &
python  ${path}/main.py 1640 &
python  ${path}/main.py 1641 &
python  ${path}/main.py 1642 &
python  ${path}/main.py 1643 &
python  ${path}/main.py 1644 &
python  ${path}/main.py 1645 &
python  ${path}/main.py 1646 &
python  ${path}/main.py 1647 &
python  ${path}/main.py 1648 &
python  ${path}/main.py 1649 &
python  ${path}/main.py 1650 &
python  ${path}/main.py 1651 &
python  ${path}/main.py 1652 &
python  ${path}/main.py 1653 &
python  ${path}/main.py 1654 &
python  ${path}/main.py 1655 &
wait
