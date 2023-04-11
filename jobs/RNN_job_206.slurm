#!/bin/bash -l
#SBATCH -J RNN_206
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7380 &
python  ${path}/main.py 7381 &
python  ${path}/main.py 7382 &
python  ${path}/main.py 7383 &
python  ${path}/main.py 7384 &
python  ${path}/main.py 7385 &
python  ${path}/main.py 7386 &
python  ${path}/main.py 7387 &
python  ${path}/main.py 7388 &
python  ${path}/main.py 7389 &
python  ${path}/main.py 7390 &
python  ${path}/main.py 7391 &
python  ${path}/main.py 7392 &
python  ${path}/main.py 7393 &
python  ${path}/main.py 7394 &
python  ${path}/main.py 7395 &
python  ${path}/main.py 7396 &
python  ${path}/main.py 7397 &
python  ${path}/main.py 7398 &
python  ${path}/main.py 7399 &
python  ${path}/main.py 7400 &
python  ${path}/main.py 7401 &
python  ${path}/main.py 7402 &
python  ${path}/main.py 7403 &
python  ${path}/main.py 7404 &
python  ${path}/main.py 7405 &
python  ${path}/main.py 7406 &
python  ${path}/main.py 7407 &
python  ${path}/main.py 7408 &
python  ${path}/main.py 7409 &
python  ${path}/main.py 7410 &
python  ${path}/main.py 7411 &
python  ${path}/main.py 7412 &
python  ${path}/main.py 7413 &
python  ${path}/main.py 7414 &
python  ${path}/main.py 7415 &
wait
