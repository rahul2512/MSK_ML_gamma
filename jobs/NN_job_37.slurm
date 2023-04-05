#!/bin/bash -l
#SBATCH -J NN_37
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
