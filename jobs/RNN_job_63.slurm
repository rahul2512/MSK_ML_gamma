#!/bin/bash -l
#SBATCH -J RNN_63
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2232 &
python  ${path}/main.py 2233 &
python  ${path}/main.py 2234 &
python  ${path}/main.py 2235 &
python  ${path}/main.py 2236 &
python  ${path}/main.py 2237 &
python  ${path}/main.py 2238 &
python  ${path}/main.py 2239 &
python  ${path}/main.py 2240 &
python  ${path}/main.py 2241 &
python  ${path}/main.py 2242 &
python  ${path}/main.py 2243 &
python  ${path}/main.py 2244 &
python  ${path}/main.py 2245 &
python  ${path}/main.py 2246 &
python  ${path}/main.py 2247 &
python  ${path}/main.py 2248 &
python  ${path}/main.py 2249 &
python  ${path}/main.py 2250 &
python  ${path}/main.py 2251 &
python  ${path}/main.py 2252 &
python  ${path}/main.py 2253 &
python  ${path}/main.py 2254 &
python  ${path}/main.py 2255 &
python  ${path}/main.py 2256 &
python  ${path}/main.py 2257 &
python  ${path}/main.py 2258 &
python  ${path}/main.py 2259 &
python  ${path}/main.py 2260 &
python  ${path}/main.py 2261 &
python  ${path}/main.py 2262 &
python  ${path}/main.py 2263 &
python  ${path}/main.py 2264 &
python  ${path}/main.py 2265 &
python  ${path}/main.py 2266 &
python  ${path}/main.py 2267 &
wait
