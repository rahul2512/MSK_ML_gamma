#!/bin/bash -l
#SBATCH -J RNN_71
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2520 &
python  ${path}/main.py 2521 &
python  ${path}/main.py 2522 &
python  ${path}/main.py 2523 &
python  ${path}/main.py 2524 &
python  ${path}/main.py 2525 &
python  ${path}/main.py 2526 &
python  ${path}/main.py 2527 &
python  ${path}/main.py 2528 &
python  ${path}/main.py 2529 &
python  ${path}/main.py 2530 &
python  ${path}/main.py 2531 &
python  ${path}/main.py 2532 &
python  ${path}/main.py 2533 &
python  ${path}/main.py 2534 &
python  ${path}/main.py 2535 &
python  ${path}/main.py 2536 &
python  ${path}/main.py 2537 &
python  ${path}/main.py 2538 &
python  ${path}/main.py 2539 &
python  ${path}/main.py 2540 &
python  ${path}/main.py 2541 &
python  ${path}/main.py 2542 &
python  ${path}/main.py 2543 &
python  ${path}/main.py 2544 &
python  ${path}/main.py 2545 &
python  ${path}/main.py 2546 &
python  ${path}/main.py 2547 &
python  ${path}/main.py 2548 &
python  ${path}/main.py 2549 &
python  ${path}/main.py 2550 &
python  ${path}/main.py 2551 &
python  ${path}/main.py 2552 &
python  ${path}/main.py 2553 &
python  ${path}/main.py 2554 &
python  ${path}/main.py 2555 &
wait
