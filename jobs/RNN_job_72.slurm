#!/bin/bash -l
#SBATCH -J RNN_72
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2556 &
python  ${path}/main.py 2557 &
python  ${path}/main.py 2558 &
python  ${path}/main.py 2559 &
python  ${path}/main.py 2560 &
python  ${path}/main.py 2561 &
python  ${path}/main.py 2562 &
python  ${path}/main.py 2563 &
python  ${path}/main.py 2564 &
python  ${path}/main.py 2565 &
python  ${path}/main.py 2566 &
python  ${path}/main.py 2567 &
python  ${path}/main.py 2568 &
python  ${path}/main.py 2569 &
python  ${path}/main.py 2570 &
python  ${path}/main.py 2571 &
python  ${path}/main.py 2572 &
python  ${path}/main.py 2573 &
python  ${path}/main.py 2574 &
python  ${path}/main.py 2575 &
python  ${path}/main.py 2576 &
python  ${path}/main.py 2577 &
python  ${path}/main.py 2578 &
python  ${path}/main.py 2579 &
python  ${path}/main.py 2580 &
python  ${path}/main.py 2581 &
python  ${path}/main.py 2582 &
python  ${path}/main.py 2583 &
python  ${path}/main.py 2584 &
python  ${path}/main.py 2585 &
python  ${path}/main.py 2586 &
python  ${path}/main.py 2587 &
python  ${path}/main.py 2588 &
python  ${path}/main.py 2589 &
python  ${path}/main.py 2590 &
python  ${path}/main.py 2591 &
wait
