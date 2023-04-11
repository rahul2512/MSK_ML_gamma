#!/bin/bash -l
#SBATCH -J RNN_69
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2448 &
python  ${path}/main.py 2449 &
python  ${path}/main.py 2450 &
python  ${path}/main.py 2451 &
python  ${path}/main.py 2452 &
python  ${path}/main.py 2453 &
python  ${path}/main.py 2454 &
python  ${path}/main.py 2455 &
python  ${path}/main.py 2456 &
python  ${path}/main.py 2457 &
python  ${path}/main.py 2458 &
python  ${path}/main.py 2459 &
python  ${path}/main.py 2460 &
python  ${path}/main.py 2461 &
python  ${path}/main.py 2462 &
python  ${path}/main.py 2463 &
python  ${path}/main.py 2464 &
python  ${path}/main.py 2465 &
python  ${path}/main.py 2466 &
python  ${path}/main.py 2467 &
python  ${path}/main.py 2468 &
python  ${path}/main.py 2469 &
python  ${path}/main.py 2470 &
python  ${path}/main.py 2471 &
python  ${path}/main.py 2472 &
python  ${path}/main.py 2473 &
python  ${path}/main.py 2474 &
python  ${path}/main.py 2475 &
python  ${path}/main.py 2476 &
python  ${path}/main.py 2477 &
python  ${path}/main.py 2478 &
python  ${path}/main.py 2479 &
python  ${path}/main.py 2480 &
python  ${path}/main.py 2481 &
python  ${path}/main.py 2482 &
python  ${path}/main.py 2483 &
wait
