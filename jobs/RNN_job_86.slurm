#!/bin/bash -l
#SBATCH -J RNN_86
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
