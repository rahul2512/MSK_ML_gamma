#!/bin/bash -l
#SBATCH -J RNN_67
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2376 &
python  ${path}/main.py 2377 &
python  ${path}/main.py 2378 &
python  ${path}/main.py 2379 &
python  ${path}/main.py 2380 &
python  ${path}/main.py 2381 &
python  ${path}/main.py 2382 &
python  ${path}/main.py 2383 &
python  ${path}/main.py 2384 &
python  ${path}/main.py 2385 &
python  ${path}/main.py 2386 &
python  ${path}/main.py 2387 &
python  ${path}/main.py 2388 &
python  ${path}/main.py 2389 &
python  ${path}/main.py 2390 &
python  ${path}/main.py 2391 &
python  ${path}/main.py 2392 &
python  ${path}/main.py 2393 &
python  ${path}/main.py 2394 &
python  ${path}/main.py 2395 &
python  ${path}/main.py 2396 &
python  ${path}/main.py 2397 &
python  ${path}/main.py 2398 &
python  ${path}/main.py 2399 &
python  ${path}/main.py 2400 &
python  ${path}/main.py 2401 &
python  ${path}/main.py 2402 &
python  ${path}/main.py 2403 &
python  ${path}/main.py 2404 &
python  ${path}/main.py 2405 &
python  ${path}/main.py 2406 &
python  ${path}/main.py 2407 &
python  ${path}/main.py 2408 &
python  ${path}/main.py 2409 &
python  ${path}/main.py 2410 &
python  ${path}/main.py 2411 &
wait
