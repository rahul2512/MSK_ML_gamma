#!/bin/bash -l
#SBATCH -J NN_38
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2664 &
python  ${path}/main.py 2665 &
python  ${path}/main.py 2666 &
python  ${path}/main.py 2667 &
python  ${path}/main.py 2668 &
python  ${path}/main.py 2669 &
python  ${path}/main.py 2670 &
python  ${path}/main.py 2671 &
python  ${path}/main.py 2672 &
python  ${path}/main.py 2673 &
python  ${path}/main.py 2674 &
python  ${path}/main.py 2675 &
python  ${path}/main.py 2676 &
python  ${path}/main.py 2677 &
python  ${path}/main.py 2678 &
python  ${path}/main.py 2679 &
python  ${path}/main.py 2680 &
python  ${path}/main.py 2681 &
python  ${path}/main.py 2682 &
python  ${path}/main.py 2683 &
python  ${path}/main.py 2684 &
python  ${path}/main.py 2685 &
python  ${path}/main.py 2686 &
python  ${path}/main.py 2687 &
python  ${path}/main.py 2688 &
python  ${path}/main.py 2689 &
python  ${path}/main.py 2690 &
python  ${path}/main.py 2691 &
python  ${path}/main.py 2692 &
python  ${path}/main.py 2693 &
python  ${path}/main.py 2694 &
python  ${path}/main.py 2695 &
python  ${path}/main.py 2696 &
python  ${path}/main.py 2697 &
python  ${path}/main.py 2698 &
python  ${path}/main.py 2699 &
python  ${path}/main.py 2700 &
python  ${path}/main.py 2701 &
python  ${path}/main.py 2702 &
python  ${path}/main.py 2703 &
python  ${path}/main.py 2704 &
python  ${path}/main.py 2705 &
python  ${path}/main.py 2706 &
python  ${path}/main.py 2707 &
python  ${path}/main.py 2708 &
python  ${path}/main.py 2709 &
python  ${path}/main.py 2710 &
python  ${path}/main.py 2711 &
python  ${path}/main.py 2712 &
python  ${path}/main.py 2713 &
python  ${path}/main.py 2714 &
python  ${path}/main.py 2715 &
python  ${path}/main.py 2716 &
python  ${path}/main.py 2717 &
python  ${path}/main.py 2718 &
python  ${path}/main.py 2719 &
python  ${path}/main.py 2720 &
python  ${path}/main.py 2721 &
python  ${path}/main.py 2722 &
python  ${path}/main.py 2723 &
python  ${path}/main.py 2724 &
python  ${path}/main.py 2725 &
python  ${path}/main.py 2726 &
python  ${path}/main.py 2727 &
python  ${path}/main.py 2728 &
python  ${path}/main.py 2729 &
python  ${path}/main.py 2730 &
python  ${path}/main.py 2731 &
python  ${path}/main.py 2732 &
python  ${path}/main.py 2733 &
python  ${path}/main.py 2734 &
python  ${path}/main.py 2735 &
wait
