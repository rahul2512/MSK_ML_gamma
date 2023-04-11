#!/bin/bash -l
#SBATCH -J RNN_52
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1836 &
python  ${path}/main.py 1837 &
python  ${path}/main.py 1838 &
python  ${path}/main.py 1839 &
python  ${path}/main.py 1840 &
python  ${path}/main.py 1841 &
python  ${path}/main.py 1842 &
python  ${path}/main.py 1843 &
python  ${path}/main.py 1844 &
python  ${path}/main.py 1845 &
python  ${path}/main.py 1846 &
python  ${path}/main.py 1847 &
python  ${path}/main.py 1848 &
python  ${path}/main.py 1849 &
python  ${path}/main.py 1850 &
python  ${path}/main.py 1851 &
python  ${path}/main.py 1852 &
python  ${path}/main.py 1853 &
python  ${path}/main.py 1854 &
python  ${path}/main.py 1855 &
python  ${path}/main.py 1856 &
python  ${path}/main.py 1857 &
python  ${path}/main.py 1858 &
python  ${path}/main.py 1859 &
python  ${path}/main.py 1860 &
python  ${path}/main.py 1861 &
python  ${path}/main.py 1862 &
python  ${path}/main.py 1863 &
python  ${path}/main.py 1864 &
python  ${path}/main.py 1865 &
python  ${path}/main.py 1866 &
python  ${path}/main.py 1867 &
python  ${path}/main.py 1868 &
python  ${path}/main.py 1869 &
python  ${path}/main.py 1870 &
python  ${path}/main.py 1871 &
wait
