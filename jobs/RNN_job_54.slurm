#!/bin/bash -l
#SBATCH -J RNN_54
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1908 &
python  ${path}/main.py 1909 &
python  ${path}/main.py 1910 &
python  ${path}/main.py 1911 &
python  ${path}/main.py 1912 &
python  ${path}/main.py 1913 &
python  ${path}/main.py 1914 &
python  ${path}/main.py 1915 &
python  ${path}/main.py 1916 &
python  ${path}/main.py 1917 &
python  ${path}/main.py 1918 &
python  ${path}/main.py 1919 &
python  ${path}/main.py 1920 &
python  ${path}/main.py 1921 &
python  ${path}/main.py 1922 &
python  ${path}/main.py 1923 &
python  ${path}/main.py 1924 &
python  ${path}/main.py 1925 &
python  ${path}/main.py 1926 &
python  ${path}/main.py 1927 &
python  ${path}/main.py 1928 &
python  ${path}/main.py 1929 &
python  ${path}/main.py 1930 &
python  ${path}/main.py 1931 &
python  ${path}/main.py 1932 &
python  ${path}/main.py 1933 &
python  ${path}/main.py 1934 &
python  ${path}/main.py 1935 &
python  ${path}/main.py 1936 &
python  ${path}/main.py 1937 &
python  ${path}/main.py 1938 &
python  ${path}/main.py 1939 &
python  ${path}/main.py 1940 &
python  ${path}/main.py 1941 &
python  ${path}/main.py 1942 &
python  ${path}/main.py 1943 &
wait
