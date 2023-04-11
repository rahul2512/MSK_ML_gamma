#!/bin/bash -l
#SBATCH -J RNN_125
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4464 &
python  ${path}/main.py 4465 &
python  ${path}/main.py 4466 &
python  ${path}/main.py 4467 &
python  ${path}/main.py 4468 &
python  ${path}/main.py 4469 &
python  ${path}/main.py 4470 &
python  ${path}/main.py 4471 &
python  ${path}/main.py 4472 &
python  ${path}/main.py 4473 &
python  ${path}/main.py 4474 &
python  ${path}/main.py 4475 &
python  ${path}/main.py 4476 &
python  ${path}/main.py 4477 &
python  ${path}/main.py 4478 &
python  ${path}/main.py 4479 &
python  ${path}/main.py 4480 &
python  ${path}/main.py 4481 &
python  ${path}/main.py 4482 &
python  ${path}/main.py 4483 &
python  ${path}/main.py 4484 &
python  ${path}/main.py 4485 &
python  ${path}/main.py 4486 &
python  ${path}/main.py 4487 &
python  ${path}/main.py 4488 &
python  ${path}/main.py 4489 &
python  ${path}/main.py 4490 &
python  ${path}/main.py 4491 &
python  ${path}/main.py 4492 &
python  ${path}/main.py 4493 &
python  ${path}/main.py 4494 &
python  ${path}/main.py 4495 &
python  ${path}/main.py 4496 &
python  ${path}/main.py 4497 &
python  ${path}/main.py 4498 &
python  ${path}/main.py 4499 &
wait
