#!/bin/bash -l
#SBATCH -J RNN_82
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2916 &
python  ${path}/main.py 2917 &
python  ${path}/main.py 2918 &
python  ${path}/main.py 2919 &
python  ${path}/main.py 2920 &
python  ${path}/main.py 2921 &
python  ${path}/main.py 2922 &
python  ${path}/main.py 2923 &
python  ${path}/main.py 2924 &
python  ${path}/main.py 2925 &
python  ${path}/main.py 2926 &
python  ${path}/main.py 2927 &
python  ${path}/main.py 2928 &
python  ${path}/main.py 2929 &
python  ${path}/main.py 2930 &
python  ${path}/main.py 2931 &
python  ${path}/main.py 2932 &
python  ${path}/main.py 2933 &
python  ${path}/main.py 2934 &
python  ${path}/main.py 2935 &
python  ${path}/main.py 2936 &
python  ${path}/main.py 2937 &
python  ${path}/main.py 2938 &
python  ${path}/main.py 2939 &
python  ${path}/main.py 2940 &
python  ${path}/main.py 2941 &
python  ${path}/main.py 2942 &
python  ${path}/main.py 2943 &
python  ${path}/main.py 2944 &
python  ${path}/main.py 2945 &
python  ${path}/main.py 2946 &
python  ${path}/main.py 2947 &
python  ${path}/main.py 2948 &
python  ${path}/main.py 2949 &
python  ${path}/main.py 2950 &
python  ${path}/main.py 2951 &
wait
