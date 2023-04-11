#!/bin/bash -l
#SBATCH -J RNN_51
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1800 &
python  ${path}/main.py 1801 &
python  ${path}/main.py 1802 &
python  ${path}/main.py 1803 &
python  ${path}/main.py 1804 &
python  ${path}/main.py 1805 &
python  ${path}/main.py 1806 &
python  ${path}/main.py 1807 &
python  ${path}/main.py 1808 &
python  ${path}/main.py 1809 &
python  ${path}/main.py 1810 &
python  ${path}/main.py 1811 &
python  ${path}/main.py 1812 &
python  ${path}/main.py 1813 &
python  ${path}/main.py 1814 &
python  ${path}/main.py 1815 &
python  ${path}/main.py 1816 &
python  ${path}/main.py 1817 &
python  ${path}/main.py 1818 &
python  ${path}/main.py 1819 &
python  ${path}/main.py 1820 &
python  ${path}/main.py 1821 &
python  ${path}/main.py 1822 &
python  ${path}/main.py 1823 &
python  ${path}/main.py 1824 &
python  ${path}/main.py 1825 &
python  ${path}/main.py 1826 &
python  ${path}/main.py 1827 &
python  ${path}/main.py 1828 &
python  ${path}/main.py 1829 &
python  ${path}/main.py 1830 &
python  ${path}/main.py 1831 &
python  ${path}/main.py 1832 &
python  ${path}/main.py 1833 &
python  ${path}/main.py 1834 &
python  ${path}/main.py 1835 &
wait
