#!/bin/bash -l
#SBATCH -J NN_42
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
python  ${path}/main.py 2988 &
python  ${path}/main.py 2989 &
python  ${path}/main.py 2990 &
python  ${path}/main.py 2991 &
python  ${path}/main.py 2992 &
python  ${path}/main.py 2993 &
python  ${path}/main.py 2994 &
python  ${path}/main.py 2995 &
python  ${path}/main.py 2996 &
python  ${path}/main.py 2997 &
python  ${path}/main.py 2998 &
python  ${path}/main.py 2999 &
python  ${path}/main.py 3000 &
python  ${path}/main.py 3001 &
python  ${path}/main.py 3002 &
python  ${path}/main.py 3003 &
python  ${path}/main.py 3004 &
python  ${path}/main.py 3005 &
python  ${path}/main.py 3006 &
python  ${path}/main.py 3007 &
python  ${path}/main.py 3008 &
python  ${path}/main.py 3009 &
python  ${path}/main.py 3010 &
python  ${path}/main.py 3011 &
python  ${path}/main.py 3012 &
python  ${path}/main.py 3013 &
python  ${path}/main.py 3014 &
python  ${path}/main.py 3015 &
python  ${path}/main.py 3016 &
python  ${path}/main.py 3017 &
python  ${path}/main.py 3018 &
python  ${path}/main.py 3019 &
python  ${path}/main.py 3020 &
python  ${path}/main.py 3021 &
python  ${path}/main.py 3022 &
python  ${path}/main.py 3023 &
wait
