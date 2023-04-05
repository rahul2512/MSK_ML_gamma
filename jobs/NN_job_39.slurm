#!/bin/bash -l
#SBATCH -J NN_39
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2736 &
python  ${path}/main.py 2737 &
python  ${path}/main.py 2738 &
python  ${path}/main.py 2739 &
python  ${path}/main.py 2740 &
python  ${path}/main.py 2741 &
python  ${path}/main.py 2742 &
python  ${path}/main.py 2743 &
python  ${path}/main.py 2744 &
python  ${path}/main.py 2745 &
python  ${path}/main.py 2746 &
python  ${path}/main.py 2747 &
python  ${path}/main.py 2748 &
python  ${path}/main.py 2749 &
python  ${path}/main.py 2750 &
python  ${path}/main.py 2751 &
python  ${path}/main.py 2752 &
python  ${path}/main.py 2753 &
python  ${path}/main.py 2754 &
python  ${path}/main.py 2755 &
python  ${path}/main.py 2756 &
python  ${path}/main.py 2757 &
python  ${path}/main.py 2758 &
python  ${path}/main.py 2759 &
python  ${path}/main.py 2760 &
python  ${path}/main.py 2761 &
python  ${path}/main.py 2762 &
python  ${path}/main.py 2763 &
python  ${path}/main.py 2764 &
python  ${path}/main.py 2765 &
python  ${path}/main.py 2766 &
python  ${path}/main.py 2767 &
python  ${path}/main.py 2768 &
python  ${path}/main.py 2769 &
python  ${path}/main.py 2770 &
python  ${path}/main.py 2771 &
python  ${path}/main.py 2772 &
python  ${path}/main.py 2773 &
python  ${path}/main.py 2774 &
python  ${path}/main.py 2775 &
python  ${path}/main.py 2776 &
python  ${path}/main.py 2777 &
python  ${path}/main.py 2778 &
python  ${path}/main.py 2779 &
python  ${path}/main.py 2780 &
python  ${path}/main.py 2781 &
python  ${path}/main.py 2782 &
python  ${path}/main.py 2783 &
python  ${path}/main.py 2784 &
python  ${path}/main.py 2785 &
python  ${path}/main.py 2786 &
python  ${path}/main.py 2787 &
python  ${path}/main.py 2788 &
python  ${path}/main.py 2789 &
python  ${path}/main.py 2790 &
python  ${path}/main.py 2791 &
python  ${path}/main.py 2792 &
python  ${path}/main.py 2793 &
python  ${path}/main.py 2794 &
python  ${path}/main.py 2795 &
python  ${path}/main.py 2796 &
python  ${path}/main.py 2797 &
python  ${path}/main.py 2798 &
python  ${path}/main.py 2799 &
python  ${path}/main.py 2800 &
python  ${path}/main.py 2801 &
python  ${path}/main.py 2802 &
python  ${path}/main.py 2803 &
python  ${path}/main.py 2804 &
python  ${path}/main.py 2805 &
python  ${path}/main.py 2806 &
python  ${path}/main.py 2807 &
wait
