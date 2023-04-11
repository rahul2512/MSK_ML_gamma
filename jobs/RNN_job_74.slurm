#!/bin/bash -l
#SBATCH -J RNN_74
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2628 &
python  ${path}/main.py 2629 &
python  ${path}/main.py 2630 &
python  ${path}/main.py 2631 &
python  ${path}/main.py 2632 &
python  ${path}/main.py 2633 &
python  ${path}/main.py 2634 &
python  ${path}/main.py 2635 &
python  ${path}/main.py 2636 &
python  ${path}/main.py 2637 &
python  ${path}/main.py 2638 &
python  ${path}/main.py 2639 &
python  ${path}/main.py 2640 &
python  ${path}/main.py 2641 &
python  ${path}/main.py 2642 &
python  ${path}/main.py 2643 &
python  ${path}/main.py 2644 &
python  ${path}/main.py 2645 &
python  ${path}/main.py 2646 &
python  ${path}/main.py 2647 &
python  ${path}/main.py 2648 &
python  ${path}/main.py 2649 &
python  ${path}/main.py 2650 &
python  ${path}/main.py 2651 &
python  ${path}/main.py 2652 &
python  ${path}/main.py 2653 &
python  ${path}/main.py 2654 &
python  ${path}/main.py 2655 &
python  ${path}/main.py 2656 &
python  ${path}/main.py 2657 &
python  ${path}/main.py 2658 &
python  ${path}/main.py 2659 &
python  ${path}/main.py 2660 &
python  ${path}/main.py 2661 &
python  ${path}/main.py 2662 &
python  ${path}/main.py 2663 &
wait
