#!/bin/bash -l
#SBATCH -J RNN_53
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1872 &
python  ${path}/main.py 1873 &
python  ${path}/main.py 1874 &
python  ${path}/main.py 1875 &
python  ${path}/main.py 1876 &
python  ${path}/main.py 1877 &
python  ${path}/main.py 1878 &
python  ${path}/main.py 1879 &
python  ${path}/main.py 1880 &
python  ${path}/main.py 1881 &
python  ${path}/main.py 1882 &
python  ${path}/main.py 1883 &
python  ${path}/main.py 1884 &
python  ${path}/main.py 1885 &
python  ${path}/main.py 1886 &
python  ${path}/main.py 1887 &
python  ${path}/main.py 1888 &
python  ${path}/main.py 1889 &
python  ${path}/main.py 1890 &
python  ${path}/main.py 1891 &
python  ${path}/main.py 1892 &
python  ${path}/main.py 1893 &
python  ${path}/main.py 1894 &
python  ${path}/main.py 1895 &
python  ${path}/main.py 1896 &
python  ${path}/main.py 1897 &
python  ${path}/main.py 1898 &
python  ${path}/main.py 1899 &
python  ${path}/main.py 1900 &
python  ${path}/main.py 1901 &
python  ${path}/main.py 1902 &
python  ${path}/main.py 1903 &
python  ${path}/main.py 1904 &
python  ${path}/main.py 1905 &
python  ${path}/main.py 1906 &
python  ${path}/main.py 1907 &
wait
