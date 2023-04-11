#!/bin/bash -l
#SBATCH -J RNN_59
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2088 &
python  ${path}/main.py 2089 &
python  ${path}/main.py 2090 &
python  ${path}/main.py 2091 &
python  ${path}/main.py 2092 &
python  ${path}/main.py 2093 &
python  ${path}/main.py 2094 &
python  ${path}/main.py 2095 &
python  ${path}/main.py 2096 &
python  ${path}/main.py 2097 &
python  ${path}/main.py 2098 &
python  ${path}/main.py 2099 &
python  ${path}/main.py 2100 &
python  ${path}/main.py 2101 &
python  ${path}/main.py 2102 &
python  ${path}/main.py 2103 &
python  ${path}/main.py 2104 &
python  ${path}/main.py 2105 &
python  ${path}/main.py 2106 &
python  ${path}/main.py 2107 &
python  ${path}/main.py 2108 &
python  ${path}/main.py 2109 &
python  ${path}/main.py 2110 &
python  ${path}/main.py 2111 &
python  ${path}/main.py 2112 &
python  ${path}/main.py 2113 &
python  ${path}/main.py 2114 &
python  ${path}/main.py 2115 &
python  ${path}/main.py 2116 &
python  ${path}/main.py 2117 &
python  ${path}/main.py 2118 &
python  ${path}/main.py 2119 &
python  ${path}/main.py 2120 &
python  ${path}/main.py 2121 &
python  ${path}/main.py 2122 &
python  ${path}/main.py 2123 &
wait
