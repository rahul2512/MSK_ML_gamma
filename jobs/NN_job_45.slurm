#!/bin/bash -l
#SBATCH -J NN_45
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3168 &
python  ${path}/main.py 3169 &
python  ${path}/main.py 3170 &
python  ${path}/main.py 3171 &
python  ${path}/main.py 3172 &
python  ${path}/main.py 3173 &
python  ${path}/main.py 3174 &
python  ${path}/main.py 3175 &
python  ${path}/main.py 3176 &
python  ${path}/main.py 3177 &
python  ${path}/main.py 3178 &
python  ${path}/main.py 3179 &
python  ${path}/main.py 3180 &
python  ${path}/main.py 3181 &
python  ${path}/main.py 3182 &
python  ${path}/main.py 3183 &
python  ${path}/main.py 3184 &
python  ${path}/main.py 3185 &
python  ${path}/main.py 3186 &
python  ${path}/main.py 3187 &
python  ${path}/main.py 3188 &
python  ${path}/main.py 3189 &
python  ${path}/main.py 3190 &
python  ${path}/main.py 3191 &
python  ${path}/main.py 3192 &
python  ${path}/main.py 3193 &
python  ${path}/main.py 3194 &
python  ${path}/main.py 3195 &
python  ${path}/main.py 3196 &
python  ${path}/main.py 3197 &
python  ${path}/main.py 3198 &
python  ${path}/main.py 3199 &
python  ${path}/main.py 3200 &
python  ${path}/main.py 3201 &
python  ${path}/main.py 3202 &
python  ${path}/main.py 3203 &
python  ${path}/main.py 3204 &
python  ${path}/main.py 3205 &
python  ${path}/main.py 3206 &
python  ${path}/main.py 3207 &
python  ${path}/main.py 3208 &
python  ${path}/main.py 3209 &
python  ${path}/main.py 3210 &
python  ${path}/main.py 3211 &
python  ${path}/main.py 3212 &
python  ${path}/main.py 3213 &
python  ${path}/main.py 3214 &
python  ${path}/main.py 3215 &
python  ${path}/main.py 3216 &
python  ${path}/main.py 3217 &
python  ${path}/main.py 3218 &
python  ${path}/main.py 3219 &
python  ${path}/main.py 3220 &
python  ${path}/main.py 3221 &
python  ${path}/main.py 3222 &
python  ${path}/main.py 3223 &
python  ${path}/main.py 3224 &
python  ${path}/main.py 3225 &
python  ${path}/main.py 3226 &
python  ${path}/main.py 3227 &
python  ${path}/main.py 3228 &
python  ${path}/main.py 3229 &
python  ${path}/main.py 3230 &
python  ${path}/main.py 3231 &
python  ${path}/main.py 3232 &
python  ${path}/main.py 3233 &
python  ${path}/main.py 3234 &
python  ${path}/main.py 3235 &
python  ${path}/main.py 3236 &
python  ${path}/main.py 3237 &
python  ${path}/main.py 3238 &
python  ${path}/main.py 3239 &
wait
