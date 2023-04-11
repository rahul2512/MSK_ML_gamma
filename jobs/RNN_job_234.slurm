#!/bin/bash -l
#SBATCH -J RNN_234
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8388 &
python  ${path}/main.py 8389 &
python  ${path}/main.py 8390 &
python  ${path}/main.py 8391 &
python  ${path}/main.py 8392 &
python  ${path}/main.py 8393 &
python  ${path}/main.py 8394 &
python  ${path}/main.py 8395 &
python  ${path}/main.py 8396 &
python  ${path}/main.py 8397 &
python  ${path}/main.py 8398 &
python  ${path}/main.py 8399 &
python  ${path}/main.py 8400 &
python  ${path}/main.py 8401 &
python  ${path}/main.py 8402 &
python  ${path}/main.py 8403 &
python  ${path}/main.py 8404 &
python  ${path}/main.py 8405 &
python  ${path}/main.py 8406 &
python  ${path}/main.py 8407 &
python  ${path}/main.py 8408 &
python  ${path}/main.py 8409 &
python  ${path}/main.py 8410 &
python  ${path}/main.py 8411 &
python  ${path}/main.py 8412 &
python  ${path}/main.py 8413 &
python  ${path}/main.py 8414 &
python  ${path}/main.py 8415 &
python  ${path}/main.py 8416 &
python  ${path}/main.py 8417 &
python  ${path}/main.py 8418 &
python  ${path}/main.py 8419 &
python  ${path}/main.py 8420 &
python  ${path}/main.py 8421 &
python  ${path}/main.py 8422 &
python  ${path}/main.py 8423 &
wait
