#!/bin/bash -l
#SBATCH -J NN_46
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3240 &
python  ${path}/main.py 3241 &
python  ${path}/main.py 3242 &
python  ${path}/main.py 3243 &
python  ${path}/main.py 3244 &
python  ${path}/main.py 3245 &
python  ${path}/main.py 3246 &
python  ${path}/main.py 3247 &
python  ${path}/main.py 3248 &
python  ${path}/main.py 3249 &
python  ${path}/main.py 3250 &
python  ${path}/main.py 3251 &
python  ${path}/main.py 3252 &
python  ${path}/main.py 3253 &
python  ${path}/main.py 3254 &
python  ${path}/main.py 3255 &
python  ${path}/main.py 3256 &
python  ${path}/main.py 3257 &
python  ${path}/main.py 3258 &
python  ${path}/main.py 3259 &
python  ${path}/main.py 3260 &
python  ${path}/main.py 3261 &
python  ${path}/main.py 3262 &
python  ${path}/main.py 3263 &
python  ${path}/main.py 3264 &
python  ${path}/main.py 3265 &
python  ${path}/main.py 3266 &
python  ${path}/main.py 3267 &
python  ${path}/main.py 3268 &
python  ${path}/main.py 3269 &
python  ${path}/main.py 3270 &
python  ${path}/main.py 3271 &
python  ${path}/main.py 3272 &
python  ${path}/main.py 3273 &
python  ${path}/main.py 3274 &
python  ${path}/main.py 3275 &
python  ${path}/main.py 3276 &
python  ${path}/main.py 3277 &
python  ${path}/main.py 3278 &
python  ${path}/main.py 3279 &
python  ${path}/main.py 3280 &
python  ${path}/main.py 3281 &
python  ${path}/main.py 3282 &
python  ${path}/main.py 3283 &
python  ${path}/main.py 3284 &
python  ${path}/main.py 3285 &
python  ${path}/main.py 3286 &
python  ${path}/main.py 3287 &
python  ${path}/main.py 3288 &
python  ${path}/main.py 3289 &
python  ${path}/main.py 3290 &
python  ${path}/main.py 3291 &
python  ${path}/main.py 3292 &
python  ${path}/main.py 3293 &
python  ${path}/main.py 3294 &
python  ${path}/main.py 3295 &
python  ${path}/main.py 3296 &
python  ${path}/main.py 3297 &
python  ${path}/main.py 3298 &
python  ${path}/main.py 3299 &
python  ${path}/main.py 3300 &
python  ${path}/main.py 3301 &
python  ${path}/main.py 3302 &
python  ${path}/main.py 3303 &
python  ${path}/main.py 3304 &
python  ${path}/main.py 3305 &
python  ${path}/main.py 3306 &
python  ${path}/main.py 3307 &
python  ${path}/main.py 3308 &
python  ${path}/main.py 3309 &
python  ${path}/main.py 3310 &
python  ${path}/main.py 3311 &
wait
