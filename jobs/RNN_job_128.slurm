#!/bin/bash -l
#SBATCH -J RNN_128
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4572 &
python  ${path}/main.py 4573 &
python  ${path}/main.py 4574 &
python  ${path}/main.py 4575 &
python  ${path}/main.py 4576 &
python  ${path}/main.py 4577 &
python  ${path}/main.py 4578 &
python  ${path}/main.py 4579 &
python  ${path}/main.py 4580 &
python  ${path}/main.py 4581 &
python  ${path}/main.py 4582 &
python  ${path}/main.py 4583 &
python  ${path}/main.py 4584 &
python  ${path}/main.py 4585 &
python  ${path}/main.py 4586 &
python  ${path}/main.py 4587 &
python  ${path}/main.py 4588 &
python  ${path}/main.py 4589 &
python  ${path}/main.py 4590 &
python  ${path}/main.py 4591 &
python  ${path}/main.py 4592 &
python  ${path}/main.py 4593 &
python  ${path}/main.py 4594 &
python  ${path}/main.py 4595 &
python  ${path}/main.py 4596 &
python  ${path}/main.py 4597 &
python  ${path}/main.py 4598 &
python  ${path}/main.py 4599 &
python  ${path}/main.py 4600 &
python  ${path}/main.py 4601 &
python  ${path}/main.py 4602 &
python  ${path}/main.py 4603 &
python  ${path}/main.py 4604 &
python  ${path}/main.py 4605 &
python  ${path}/main.py 4606 &
python  ${path}/main.py 4607 &
wait
