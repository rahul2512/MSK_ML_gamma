#!/bin/bash -l
#SBATCH -J RNN_203
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
wait
