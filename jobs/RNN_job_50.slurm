#!/bin/bash -l
#SBATCH -J RNN_50
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1764 &
python  ${path}/main.py 1765 &
python  ${path}/main.py 1766 &
python  ${path}/main.py 1767 &
python  ${path}/main.py 1768 &
python  ${path}/main.py 1769 &
python  ${path}/main.py 1770 &
python  ${path}/main.py 1771 &
python  ${path}/main.py 1772 &
python  ${path}/main.py 1773 &
python  ${path}/main.py 1774 &
python  ${path}/main.py 1775 &
python  ${path}/main.py 1776 &
python  ${path}/main.py 1777 &
python  ${path}/main.py 1778 &
python  ${path}/main.py 1779 &
python  ${path}/main.py 1780 &
python  ${path}/main.py 1781 &
python  ${path}/main.py 1782 &
python  ${path}/main.py 1783 &
python  ${path}/main.py 1784 &
python  ${path}/main.py 1785 &
python  ${path}/main.py 1786 &
python  ${path}/main.py 1787 &
python  ${path}/main.py 1788 &
python  ${path}/main.py 1789 &
python  ${path}/main.py 1790 &
python  ${path}/main.py 1791 &
python  ${path}/main.py 1792 &
python  ${path}/main.py 1793 &
python  ${path}/main.py 1794 &
python  ${path}/main.py 1795 &
python  ${path}/main.py 1796 &
python  ${path}/main.py 1797 &
python  ${path}/main.py 1798 &
python  ${path}/main.py 1799 &
wait
