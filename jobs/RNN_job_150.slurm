#!/bin/bash -l
#SBATCH -J RNN_150
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5364 &
python  ${path}/main.py 5365 &
python  ${path}/main.py 5366 &
python  ${path}/main.py 5367 &
python  ${path}/main.py 5368 &
python  ${path}/main.py 5369 &
python  ${path}/main.py 5370 &
python  ${path}/main.py 5371 &
python  ${path}/main.py 5372 &
python  ${path}/main.py 5373 &
python  ${path}/main.py 5374 &
python  ${path}/main.py 5375 &
python  ${path}/main.py 5376 &
python  ${path}/main.py 5377 &
python  ${path}/main.py 5378 &
python  ${path}/main.py 5379 &
python  ${path}/main.py 5380 &
python  ${path}/main.py 5381 &
python  ${path}/main.py 5382 &
python  ${path}/main.py 5383 &
python  ${path}/main.py 5384 &
python  ${path}/main.py 5385 &
python  ${path}/main.py 5386 &
python  ${path}/main.py 5387 &
python  ${path}/main.py 5388 &
python  ${path}/main.py 5389 &
python  ${path}/main.py 5390 &
python  ${path}/main.py 5391 &
python  ${path}/main.py 5392 &
python  ${path}/main.py 5393 &
python  ${path}/main.py 5394 &
python  ${path}/main.py 5395 &
python  ${path}/main.py 5396 &
python  ${path}/main.py 5397 &
python  ${path}/main.py 5398 &
python  ${path}/main.py 5399 &
wait
