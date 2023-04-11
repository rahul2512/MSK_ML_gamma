#!/bin/bash -l
#SBATCH -J RNN_83
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2952 &
python  ${path}/main.py 2953 &
python  ${path}/main.py 2954 &
python  ${path}/main.py 2955 &
python  ${path}/main.py 2956 &
python  ${path}/main.py 2957 &
python  ${path}/main.py 2958 &
python  ${path}/main.py 2959 &
python  ${path}/main.py 2960 &
python  ${path}/main.py 2961 &
python  ${path}/main.py 2962 &
python  ${path}/main.py 2963 &
python  ${path}/main.py 2964 &
python  ${path}/main.py 2965 &
python  ${path}/main.py 2966 &
python  ${path}/main.py 2967 &
python  ${path}/main.py 2968 &
python  ${path}/main.py 2969 &
python  ${path}/main.py 2970 &
python  ${path}/main.py 2971 &
python  ${path}/main.py 2972 &
python  ${path}/main.py 2973 &
python  ${path}/main.py 2974 &
python  ${path}/main.py 2975 &
python  ${path}/main.py 2976 &
python  ${path}/main.py 2977 &
python  ${path}/main.py 2978 &
python  ${path}/main.py 2979 &
python  ${path}/main.py 2980 &
python  ${path}/main.py 2981 &
python  ${path}/main.py 2982 &
python  ${path}/main.py 2983 &
python  ${path}/main.py 2984 &
python  ${path}/main.py 2985 &
python  ${path}/main.py 2986 &
python  ${path}/main.py 2987 &
wait
