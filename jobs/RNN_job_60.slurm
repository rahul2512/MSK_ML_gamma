#!/bin/bash -l
#SBATCH -J RNN_60
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2124 &
python  ${path}/main.py 2125 &
python  ${path}/main.py 2126 &
python  ${path}/main.py 2127 &
python  ${path}/main.py 2128 &
python  ${path}/main.py 2129 &
python  ${path}/main.py 2130 &
python  ${path}/main.py 2131 &
python  ${path}/main.py 2132 &
python  ${path}/main.py 2133 &
python  ${path}/main.py 2134 &
python  ${path}/main.py 2135 &
python  ${path}/main.py 2136 &
python  ${path}/main.py 2137 &
python  ${path}/main.py 2138 &
python  ${path}/main.py 2139 &
python  ${path}/main.py 2140 &
python  ${path}/main.py 2141 &
python  ${path}/main.py 2142 &
python  ${path}/main.py 2143 &
python  ${path}/main.py 2144 &
python  ${path}/main.py 2145 &
python  ${path}/main.py 2146 &
python  ${path}/main.py 2147 &
python  ${path}/main.py 2148 &
python  ${path}/main.py 2149 &
python  ${path}/main.py 2150 &
python  ${path}/main.py 2151 &
python  ${path}/main.py 2152 &
python  ${path}/main.py 2153 &
python  ${path}/main.py 2154 &
python  ${path}/main.py 2155 &
python  ${path}/main.py 2156 &
python  ${path}/main.py 2157 &
python  ${path}/main.py 2158 &
python  ${path}/main.py 2159 &
wait
