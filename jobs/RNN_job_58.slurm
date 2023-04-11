#!/bin/bash -l
#SBATCH -J RNN_58
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2052 &
python  ${path}/main.py 2053 &
python  ${path}/main.py 2054 &
python  ${path}/main.py 2055 &
python  ${path}/main.py 2056 &
python  ${path}/main.py 2057 &
python  ${path}/main.py 2058 &
python  ${path}/main.py 2059 &
python  ${path}/main.py 2060 &
python  ${path}/main.py 2061 &
python  ${path}/main.py 2062 &
python  ${path}/main.py 2063 &
python  ${path}/main.py 2064 &
python  ${path}/main.py 2065 &
python  ${path}/main.py 2066 &
python  ${path}/main.py 2067 &
python  ${path}/main.py 2068 &
python  ${path}/main.py 2069 &
python  ${path}/main.py 2070 &
python  ${path}/main.py 2071 &
python  ${path}/main.py 2072 &
python  ${path}/main.py 2073 &
python  ${path}/main.py 2074 &
python  ${path}/main.py 2075 &
python  ${path}/main.py 2076 &
python  ${path}/main.py 2077 &
python  ${path}/main.py 2078 &
python  ${path}/main.py 2079 &
python  ${path}/main.py 2080 &
python  ${path}/main.py 2081 &
python  ${path}/main.py 2082 &
python  ${path}/main.py 2083 &
python  ${path}/main.py 2084 &
python  ${path}/main.py 2085 &
python  ${path}/main.py 2086 &
python  ${path}/main.py 2087 &
wait
