#!/bin/bash -l
#SBATCH -J RNN_151
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5400 &
python  ${path}/main.py 5401 &
python  ${path}/main.py 5402 &
python  ${path}/main.py 5403 &
python  ${path}/main.py 5404 &
python  ${path}/main.py 5405 &
python  ${path}/main.py 5406 &
python  ${path}/main.py 5407 &
python  ${path}/main.py 5408 &
python  ${path}/main.py 5409 &
python  ${path}/main.py 5410 &
python  ${path}/main.py 5411 &
python  ${path}/main.py 5412 &
python  ${path}/main.py 5413 &
python  ${path}/main.py 5414 &
python  ${path}/main.py 5415 &
python  ${path}/main.py 5416 &
python  ${path}/main.py 5417 &
python  ${path}/main.py 5418 &
python  ${path}/main.py 5419 &
python  ${path}/main.py 5420 &
python  ${path}/main.py 5421 &
python  ${path}/main.py 5422 &
python  ${path}/main.py 5423 &
python  ${path}/main.py 5424 &
python  ${path}/main.py 5425 &
python  ${path}/main.py 5426 &
python  ${path}/main.py 5427 &
python  ${path}/main.py 5428 &
python  ${path}/main.py 5429 &
python  ${path}/main.py 5430 &
python  ${path}/main.py 5431 &
python  ${path}/main.py 5432 &
python  ${path}/main.py 5433 &
python  ${path}/main.py 5434 &
python  ${path}/main.py 5435 &
wait
