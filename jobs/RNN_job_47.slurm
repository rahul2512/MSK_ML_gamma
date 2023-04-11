#!/bin/bash -l
#SBATCH -J RNN_47
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1656 &
python  ${path}/main.py 1657 &
python  ${path}/main.py 1658 &
python  ${path}/main.py 1659 &
python  ${path}/main.py 1660 &
python  ${path}/main.py 1661 &
python  ${path}/main.py 1662 &
python  ${path}/main.py 1663 &
python  ${path}/main.py 1664 &
python  ${path}/main.py 1665 &
python  ${path}/main.py 1666 &
python  ${path}/main.py 1667 &
python  ${path}/main.py 1668 &
python  ${path}/main.py 1669 &
python  ${path}/main.py 1670 &
python  ${path}/main.py 1671 &
python  ${path}/main.py 1672 &
python  ${path}/main.py 1673 &
python  ${path}/main.py 1674 &
python  ${path}/main.py 1675 &
python  ${path}/main.py 1676 &
python  ${path}/main.py 1677 &
python  ${path}/main.py 1678 &
python  ${path}/main.py 1679 &
python  ${path}/main.py 1680 &
python  ${path}/main.py 1681 &
python  ${path}/main.py 1682 &
python  ${path}/main.py 1683 &
python  ${path}/main.py 1684 &
python  ${path}/main.py 1685 &
python  ${path}/main.py 1686 &
python  ${path}/main.py 1687 &
python  ${path}/main.py 1688 &
python  ${path}/main.py 1689 &
python  ${path}/main.py 1690 &
python  ${path}/main.py 1691 &
wait
