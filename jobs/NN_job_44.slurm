#!/bin/bash -l
#SBATCH -J NN_44
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3096 &
python  ${path}/main.py 3097 &
python  ${path}/main.py 3098 &
python  ${path}/main.py 3099 &
python  ${path}/main.py 3100 &
python  ${path}/main.py 3101 &
python  ${path}/main.py 3102 &
python  ${path}/main.py 3103 &
python  ${path}/main.py 3104 &
python  ${path}/main.py 3105 &
python  ${path}/main.py 3106 &
python  ${path}/main.py 3107 &
python  ${path}/main.py 3108 &
python  ${path}/main.py 3109 &
python  ${path}/main.py 3110 &
python  ${path}/main.py 3111 &
python  ${path}/main.py 3112 &
python  ${path}/main.py 3113 &
python  ${path}/main.py 3114 &
python  ${path}/main.py 3115 &
python  ${path}/main.py 3116 &
python  ${path}/main.py 3117 &
python  ${path}/main.py 3118 &
python  ${path}/main.py 3119 &
python  ${path}/main.py 3120 &
python  ${path}/main.py 3121 &
python  ${path}/main.py 3122 &
python  ${path}/main.py 3123 &
python  ${path}/main.py 3124 &
python  ${path}/main.py 3125 &
python  ${path}/main.py 3126 &
python  ${path}/main.py 3127 &
python  ${path}/main.py 3128 &
python  ${path}/main.py 3129 &
python  ${path}/main.py 3130 &
python  ${path}/main.py 3131 &
python  ${path}/main.py 3132 &
python  ${path}/main.py 3133 &
python  ${path}/main.py 3134 &
python  ${path}/main.py 3135 &
python  ${path}/main.py 3136 &
python  ${path}/main.py 3137 &
python  ${path}/main.py 3138 &
python  ${path}/main.py 3139 &
python  ${path}/main.py 3140 &
python  ${path}/main.py 3141 &
python  ${path}/main.py 3142 &
python  ${path}/main.py 3143 &
python  ${path}/main.py 3144 &
python  ${path}/main.py 3145 &
python  ${path}/main.py 3146 &
python  ${path}/main.py 3147 &
python  ${path}/main.py 3148 &
python  ${path}/main.py 3149 &
python  ${path}/main.py 3150 &
python  ${path}/main.py 3151 &
python  ${path}/main.py 3152 &
python  ${path}/main.py 3153 &
python  ${path}/main.py 3154 &
python  ${path}/main.py 3155 &
python  ${path}/main.py 3156 &
python  ${path}/main.py 3157 &
python  ${path}/main.py 3158 &
python  ${path}/main.py 3159 &
python  ${path}/main.py 3160 &
python  ${path}/main.py 3161 &
python  ${path}/main.py 3162 &
python  ${path}/main.py 3163 &
python  ${path}/main.py 3164 &
python  ${path}/main.py 3165 &
python  ${path}/main.py 3166 &
python  ${path}/main.py 3167 &
wait
