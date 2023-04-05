#!/bin/bash -l
#SBATCH -J NN_43
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3024 &
python  ${path}/main.py 3025 &
python  ${path}/main.py 3026 &
python  ${path}/main.py 3027 &
python  ${path}/main.py 3028 &
python  ${path}/main.py 3029 &
python  ${path}/main.py 3030 &
python  ${path}/main.py 3031 &
python  ${path}/main.py 3032 &
python  ${path}/main.py 3033 &
python  ${path}/main.py 3034 &
python  ${path}/main.py 3035 &
python  ${path}/main.py 3036 &
python  ${path}/main.py 3037 &
python  ${path}/main.py 3038 &
python  ${path}/main.py 3039 &
python  ${path}/main.py 3040 &
python  ${path}/main.py 3041 &
python  ${path}/main.py 3042 &
python  ${path}/main.py 3043 &
python  ${path}/main.py 3044 &
python  ${path}/main.py 3045 &
python  ${path}/main.py 3046 &
python  ${path}/main.py 3047 &
python  ${path}/main.py 3048 &
python  ${path}/main.py 3049 &
python  ${path}/main.py 3050 &
python  ${path}/main.py 3051 &
python  ${path}/main.py 3052 &
python  ${path}/main.py 3053 &
python  ${path}/main.py 3054 &
python  ${path}/main.py 3055 &
python  ${path}/main.py 3056 &
python  ${path}/main.py 3057 &
python  ${path}/main.py 3058 &
python  ${path}/main.py 3059 &
python  ${path}/main.py 3060 &
python  ${path}/main.py 3061 &
python  ${path}/main.py 3062 &
python  ${path}/main.py 3063 &
python  ${path}/main.py 3064 &
python  ${path}/main.py 3065 &
python  ${path}/main.py 3066 &
python  ${path}/main.py 3067 &
python  ${path}/main.py 3068 &
python  ${path}/main.py 3069 &
python  ${path}/main.py 3070 &
python  ${path}/main.py 3071 &
python  ${path}/main.py 3072 &
python  ${path}/main.py 3073 &
python  ${path}/main.py 3074 &
python  ${path}/main.py 3075 &
python  ${path}/main.py 3076 &
python  ${path}/main.py 3077 &
python  ${path}/main.py 3078 &
python  ${path}/main.py 3079 &
python  ${path}/main.py 3080 &
python  ${path}/main.py 3081 &
python  ${path}/main.py 3082 &
python  ${path}/main.py 3083 &
python  ${path}/main.py 3084 &
python  ${path}/main.py 3085 &
python  ${path}/main.py 3086 &
python  ${path}/main.py 3087 &
python  ${path}/main.py 3088 &
python  ${path}/main.py 3089 &
python  ${path}/main.py 3090 &
python  ${path}/main.py 3091 &
python  ${path}/main.py 3092 &
python  ${path}/main.py 3093 &
python  ${path}/main.py 3094 &
python  ${path}/main.py 3095 &
wait
