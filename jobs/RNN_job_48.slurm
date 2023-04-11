#!/bin/bash -l
#SBATCH -J RNN_48
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1692 &
python  ${path}/main.py 1693 &
python  ${path}/main.py 1694 &
python  ${path}/main.py 1695 &
python  ${path}/main.py 1696 &
python  ${path}/main.py 1697 &
python  ${path}/main.py 1698 &
python  ${path}/main.py 1699 &
python  ${path}/main.py 1700 &
python  ${path}/main.py 1701 &
python  ${path}/main.py 1702 &
python  ${path}/main.py 1703 &
python  ${path}/main.py 1704 &
python  ${path}/main.py 1705 &
python  ${path}/main.py 1706 &
python  ${path}/main.py 1707 &
python  ${path}/main.py 1708 &
python  ${path}/main.py 1709 &
python  ${path}/main.py 1710 &
python  ${path}/main.py 1711 &
python  ${path}/main.py 1712 &
python  ${path}/main.py 1713 &
python  ${path}/main.py 1714 &
python  ${path}/main.py 1715 &
python  ${path}/main.py 1716 &
python  ${path}/main.py 1717 &
python  ${path}/main.py 1718 &
python  ${path}/main.py 1719 &
python  ${path}/main.py 1720 &
python  ${path}/main.py 1721 &
python  ${path}/main.py 1722 &
python  ${path}/main.py 1723 &
python  ${path}/main.py 1724 &
python  ${path}/main.py 1725 &
python  ${path}/main.py 1726 &
python  ${path}/main.py 1727 &
wait
