#!/bin/bash -l
#SBATCH -J RNN_123
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4392 &
python  ${path}/main.py 4393 &
python  ${path}/main.py 4394 &
python  ${path}/main.py 4395 &
python  ${path}/main.py 4396 &
python  ${path}/main.py 4397 &
python  ${path}/main.py 4398 &
python  ${path}/main.py 4399 &
python  ${path}/main.py 4400 &
python  ${path}/main.py 4401 &
python  ${path}/main.py 4402 &
python  ${path}/main.py 4403 &
python  ${path}/main.py 4404 &
python  ${path}/main.py 4405 &
python  ${path}/main.py 4406 &
python  ${path}/main.py 4407 &
python  ${path}/main.py 4408 &
python  ${path}/main.py 4409 &
python  ${path}/main.py 4410 &
python  ${path}/main.py 4411 &
python  ${path}/main.py 4412 &
python  ${path}/main.py 4413 &
python  ${path}/main.py 4414 &
python  ${path}/main.py 4415 &
python  ${path}/main.py 4416 &
python  ${path}/main.py 4417 &
python  ${path}/main.py 4418 &
python  ${path}/main.py 4419 &
python  ${path}/main.py 4420 &
python  ${path}/main.py 4421 &
python  ${path}/main.py 4422 &
python  ${path}/main.py 4423 &
python  ${path}/main.py 4424 &
python  ${path}/main.py 4425 &
python  ${path}/main.py 4426 &
python  ${path}/main.py 4427 &
wait
