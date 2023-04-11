#!/bin/bash -l
#SBATCH -J RNN_73
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2592 &
python  ${path}/main.py 2593 &
python  ${path}/main.py 2594 &
python  ${path}/main.py 2595 &
python  ${path}/main.py 2596 &
python  ${path}/main.py 2597 &
python  ${path}/main.py 2598 &
python  ${path}/main.py 2599 &
python  ${path}/main.py 2600 &
python  ${path}/main.py 2601 &
python  ${path}/main.py 2602 &
python  ${path}/main.py 2603 &
python  ${path}/main.py 2604 &
python  ${path}/main.py 2605 &
python  ${path}/main.py 2606 &
python  ${path}/main.py 2607 &
python  ${path}/main.py 2608 &
python  ${path}/main.py 2609 &
python  ${path}/main.py 2610 &
python  ${path}/main.py 2611 &
python  ${path}/main.py 2612 &
python  ${path}/main.py 2613 &
python  ${path}/main.py 2614 &
python  ${path}/main.py 2615 &
python  ${path}/main.py 2616 &
python  ${path}/main.py 2617 &
python  ${path}/main.py 2618 &
python  ${path}/main.py 2619 &
python  ${path}/main.py 2620 &
python  ${path}/main.py 2621 &
python  ${path}/main.py 2622 &
python  ${path}/main.py 2623 &
python  ${path}/main.py 2624 &
python  ${path}/main.py 2625 &
python  ${path}/main.py 2626 &
python  ${path}/main.py 2627 &
wait
