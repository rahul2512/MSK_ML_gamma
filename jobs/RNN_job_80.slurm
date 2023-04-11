#!/bin/bash -l
#SBATCH -J RNN_80
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2844 &
python  ${path}/main.py 2845 &
python  ${path}/main.py 2846 &
python  ${path}/main.py 2847 &
python  ${path}/main.py 2848 &
python  ${path}/main.py 2849 &
python  ${path}/main.py 2850 &
python  ${path}/main.py 2851 &
python  ${path}/main.py 2852 &
python  ${path}/main.py 2853 &
python  ${path}/main.py 2854 &
python  ${path}/main.py 2855 &
python  ${path}/main.py 2856 &
python  ${path}/main.py 2857 &
python  ${path}/main.py 2858 &
python  ${path}/main.py 2859 &
python  ${path}/main.py 2860 &
python  ${path}/main.py 2861 &
python  ${path}/main.py 2862 &
python  ${path}/main.py 2863 &
python  ${path}/main.py 2864 &
python  ${path}/main.py 2865 &
python  ${path}/main.py 2866 &
python  ${path}/main.py 2867 &
python  ${path}/main.py 2868 &
python  ${path}/main.py 2869 &
python  ${path}/main.py 2870 &
python  ${path}/main.py 2871 &
python  ${path}/main.py 2872 &
python  ${path}/main.py 2873 &
python  ${path}/main.py 2874 &
python  ${path}/main.py 2875 &
python  ${path}/main.py 2876 &
python  ${path}/main.py 2877 &
python  ${path}/main.py 2878 &
python  ${path}/main.py 2879 &
wait
