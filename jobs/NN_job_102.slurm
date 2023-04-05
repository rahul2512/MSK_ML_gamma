#!/bin/bash -l
#SBATCH -J NN_102
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7272 &
python  ${path}/main.py 7273 &
python  ${path}/main.py 7274 &
python  ${path}/main.py 7275 &
python  ${path}/main.py 7276 &
python  ${path}/main.py 7277 &
python  ${path}/main.py 7278 &
python  ${path}/main.py 7279 &
python  ${path}/main.py 7280 &
python  ${path}/main.py 7281 &
python  ${path}/main.py 7282 &
python  ${path}/main.py 7283 &
python  ${path}/main.py 7284 &
python  ${path}/main.py 7285 &
python  ${path}/main.py 7286 &
python  ${path}/main.py 7287 &
python  ${path}/main.py 7288 &
python  ${path}/main.py 7289 &
python  ${path}/main.py 7290 &
python  ${path}/main.py 7291 &
python  ${path}/main.py 7292 &
python  ${path}/main.py 7293 &
python  ${path}/main.py 7294 &
python  ${path}/main.py 7295 &
python  ${path}/main.py 7296 &
python  ${path}/main.py 7297 &
python  ${path}/main.py 7298 &
python  ${path}/main.py 7299 &
python  ${path}/main.py 7300 &
python  ${path}/main.py 7301 &
python  ${path}/main.py 7302 &
python  ${path}/main.py 7303 &
python  ${path}/main.py 7304 &
python  ${path}/main.py 7305 &
python  ${path}/main.py 7306 &
python  ${path}/main.py 7307 &
python  ${path}/main.py 7308 &
python  ${path}/main.py 7309 &
python  ${path}/main.py 7310 &
python  ${path}/main.py 7311 &
python  ${path}/main.py 7312 &
python  ${path}/main.py 7313 &
python  ${path}/main.py 7314 &
python  ${path}/main.py 7315 &
python  ${path}/main.py 7316 &
python  ${path}/main.py 7317 &
python  ${path}/main.py 7318 &
python  ${path}/main.py 7319 &
python  ${path}/main.py 7320 &
python  ${path}/main.py 7321 &
python  ${path}/main.py 7322 &
python  ${path}/main.py 7323 &
python  ${path}/main.py 7324 &
python  ${path}/main.py 7325 &
python  ${path}/main.py 7326 &
python  ${path}/main.py 7327 &
python  ${path}/main.py 7328 &
python  ${path}/main.py 7329 &
python  ${path}/main.py 7330 &
python  ${path}/main.py 7331 &
python  ${path}/main.py 7332 &
python  ${path}/main.py 7333 &
python  ${path}/main.py 7334 &
python  ${path}/main.py 7335 &
python  ${path}/main.py 7336 &
python  ${path}/main.py 7337 &
python  ${path}/main.py 7338 &
python  ${path}/main.py 7339 &
python  ${path}/main.py 7340 &
python  ${path}/main.py 7341 &
python  ${path}/main.py 7342 &
python  ${path}/main.py 7343 &
wait
