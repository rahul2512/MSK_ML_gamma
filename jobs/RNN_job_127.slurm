#!/bin/bash -l
#SBATCH -J RNN_127
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4536 &
python  ${path}/main.py 4537 &
python  ${path}/main.py 4538 &
python  ${path}/main.py 4539 &
python  ${path}/main.py 4540 &
python  ${path}/main.py 4541 &
python  ${path}/main.py 4542 &
python  ${path}/main.py 4543 &
python  ${path}/main.py 4544 &
python  ${path}/main.py 4545 &
python  ${path}/main.py 4546 &
python  ${path}/main.py 4547 &
python  ${path}/main.py 4548 &
python  ${path}/main.py 4549 &
python  ${path}/main.py 4550 &
python  ${path}/main.py 4551 &
python  ${path}/main.py 4552 &
python  ${path}/main.py 4553 &
python  ${path}/main.py 4554 &
python  ${path}/main.py 4555 &
python  ${path}/main.py 4556 &
python  ${path}/main.py 4557 &
python  ${path}/main.py 4558 &
python  ${path}/main.py 4559 &
python  ${path}/main.py 4560 &
python  ${path}/main.py 4561 &
python  ${path}/main.py 4562 &
python  ${path}/main.py 4563 &
python  ${path}/main.py 4564 &
python  ${path}/main.py 4565 &
python  ${path}/main.py 4566 &
python  ${path}/main.py 4567 &
python  ${path}/main.py 4568 &
python  ${path}/main.py 4569 &
python  ${path}/main.py 4570 &
python  ${path}/main.py 4571 &
wait
