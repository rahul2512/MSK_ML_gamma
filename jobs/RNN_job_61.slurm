#!/bin/bash -l
#SBATCH -J RNN_61
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2160 &
python  ${path}/main.py 2161 &
python  ${path}/main.py 2162 &
python  ${path}/main.py 2163 &
python  ${path}/main.py 2164 &
python  ${path}/main.py 2165 &
python  ${path}/main.py 2166 &
python  ${path}/main.py 2167 &
python  ${path}/main.py 2168 &
python  ${path}/main.py 2169 &
python  ${path}/main.py 2170 &
python  ${path}/main.py 2171 &
python  ${path}/main.py 2172 &
python  ${path}/main.py 2173 &
python  ${path}/main.py 2174 &
python  ${path}/main.py 2175 &
python  ${path}/main.py 2176 &
python  ${path}/main.py 2177 &
python  ${path}/main.py 2178 &
python  ${path}/main.py 2179 &
python  ${path}/main.py 2180 &
python  ${path}/main.py 2181 &
python  ${path}/main.py 2182 &
python  ${path}/main.py 2183 &
python  ${path}/main.py 2184 &
python  ${path}/main.py 2185 &
python  ${path}/main.py 2186 &
python  ${path}/main.py 2187 &
python  ${path}/main.py 2188 &
python  ${path}/main.py 2189 &
python  ${path}/main.py 2190 &
python  ${path}/main.py 2191 &
python  ${path}/main.py 2192 &
python  ${path}/main.py 2193 &
python  ${path}/main.py 2194 &
python  ${path}/main.py 2195 &
wait
