#!/bin/bash -l
#SBATCH -J RNN_122
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4356 &
python  ${path}/main.py 4357 &
python  ${path}/main.py 4358 &
python  ${path}/main.py 4359 &
python  ${path}/main.py 4360 &
python  ${path}/main.py 4361 &
python  ${path}/main.py 4362 &
python  ${path}/main.py 4363 &
python  ${path}/main.py 4364 &
python  ${path}/main.py 4365 &
python  ${path}/main.py 4366 &
python  ${path}/main.py 4367 &
python  ${path}/main.py 4368 &
python  ${path}/main.py 4369 &
python  ${path}/main.py 4370 &
python  ${path}/main.py 4371 &
python  ${path}/main.py 4372 &
python  ${path}/main.py 4373 &
python  ${path}/main.py 4374 &
python  ${path}/main.py 4375 &
python  ${path}/main.py 4376 &
python  ${path}/main.py 4377 &
python  ${path}/main.py 4378 &
python  ${path}/main.py 4379 &
python  ${path}/main.py 4380 &
python  ${path}/main.py 4381 &
python  ${path}/main.py 4382 &
python  ${path}/main.py 4383 &
python  ${path}/main.py 4384 &
python  ${path}/main.py 4385 &
python  ${path}/main.py 4386 &
python  ${path}/main.py 4387 &
python  ${path}/main.py 4388 &
python  ${path}/main.py 4389 &
python  ${path}/main.py 4390 &
python  ${path}/main.py 4391 &
wait
