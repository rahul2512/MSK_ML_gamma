#!/bin/bash -l
#SBATCH -J RNN_79
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2808 &
python  ${path}/main.py 2809 &
python  ${path}/main.py 2810 &
python  ${path}/main.py 2811 &
python  ${path}/main.py 2812 &
python  ${path}/main.py 2813 &
python  ${path}/main.py 2814 &
python  ${path}/main.py 2815 &
python  ${path}/main.py 2816 &
python  ${path}/main.py 2817 &
python  ${path}/main.py 2818 &
python  ${path}/main.py 2819 &
python  ${path}/main.py 2820 &
python  ${path}/main.py 2821 &
python  ${path}/main.py 2822 &
python  ${path}/main.py 2823 &
python  ${path}/main.py 2824 &
python  ${path}/main.py 2825 &
python  ${path}/main.py 2826 &
python  ${path}/main.py 2827 &
python  ${path}/main.py 2828 &
python  ${path}/main.py 2829 &
python  ${path}/main.py 2830 &
python  ${path}/main.py 2831 &
python  ${path}/main.py 2832 &
python  ${path}/main.py 2833 &
python  ${path}/main.py 2834 &
python  ${path}/main.py 2835 &
python  ${path}/main.py 2836 &
python  ${path}/main.py 2837 &
python  ${path}/main.py 2838 &
python  ${path}/main.py 2839 &
python  ${path}/main.py 2840 &
python  ${path}/main.py 2841 &
python  ${path}/main.py 2842 &
python  ${path}/main.py 2843 &
wait
