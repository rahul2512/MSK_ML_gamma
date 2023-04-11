#!/bin/bash -l
#SBATCH -J RNN_204
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
