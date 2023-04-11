#!/bin/bash -l
#SBATCH -J RNN_78
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
