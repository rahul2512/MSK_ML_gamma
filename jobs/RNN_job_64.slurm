#!/bin/bash -l
#SBATCH -J RNN_64
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2268 &
python  ${path}/main.py 2269 &
python  ${path}/main.py 2270 &
python  ${path}/main.py 2271 &
python  ${path}/main.py 2272 &
python  ${path}/main.py 2273 &
python  ${path}/main.py 2274 &
python  ${path}/main.py 2275 &
python  ${path}/main.py 2276 &
python  ${path}/main.py 2277 &
python  ${path}/main.py 2278 &
python  ${path}/main.py 2279 &
python  ${path}/main.py 2280 &
python  ${path}/main.py 2281 &
python  ${path}/main.py 2282 &
python  ${path}/main.py 2283 &
python  ${path}/main.py 2284 &
python  ${path}/main.py 2285 &
python  ${path}/main.py 2286 &
python  ${path}/main.py 2287 &
python  ${path}/main.py 2288 &
python  ${path}/main.py 2289 &
python  ${path}/main.py 2290 &
python  ${path}/main.py 2291 &
python  ${path}/main.py 2292 &
python  ${path}/main.py 2293 &
python  ${path}/main.py 2294 &
python  ${path}/main.py 2295 &
python  ${path}/main.py 2296 &
python  ${path}/main.py 2297 &
python  ${path}/main.py 2298 &
python  ${path}/main.py 2299 &
python  ${path}/main.py 2300 &
python  ${path}/main.py 2301 &
python  ${path}/main.py 2302 &
python  ${path}/main.py 2303 &
wait
