#!/bin/bash -l
#SBATCH -J RNN_66
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2340 &
python  ${path}/main.py 2341 &
python  ${path}/main.py 2342 &
python  ${path}/main.py 2343 &
python  ${path}/main.py 2344 &
python  ${path}/main.py 2345 &
python  ${path}/main.py 2346 &
python  ${path}/main.py 2347 &
python  ${path}/main.py 2348 &
python  ${path}/main.py 2349 &
python  ${path}/main.py 2350 &
python  ${path}/main.py 2351 &
python  ${path}/main.py 2352 &
python  ${path}/main.py 2353 &
python  ${path}/main.py 2354 &
python  ${path}/main.py 2355 &
python  ${path}/main.py 2356 &
python  ${path}/main.py 2357 &
python  ${path}/main.py 2358 &
python  ${path}/main.py 2359 &
python  ${path}/main.py 2360 &
python  ${path}/main.py 2361 &
python  ${path}/main.py 2362 &
python  ${path}/main.py 2363 &
python  ${path}/main.py 2364 &
python  ${path}/main.py 2365 &
python  ${path}/main.py 2366 &
python  ${path}/main.py 2367 &
python  ${path}/main.py 2368 &
python  ${path}/main.py 2369 &
python  ${path}/main.py 2370 &
python  ${path}/main.py 2371 &
python  ${path}/main.py 2372 &
python  ${path}/main.py 2373 &
python  ${path}/main.py 2374 &
python  ${path}/main.py 2375 &
wait
