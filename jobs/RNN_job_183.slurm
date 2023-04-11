#!/bin/bash -l
#SBATCH -J RNN_183
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6552 &
python  ${path}/main.py 6553 &
python  ${path}/main.py 6554 &
python  ${path}/main.py 6555 &
python  ${path}/main.py 6556 &
python  ${path}/main.py 6557 &
python  ${path}/main.py 6558 &
python  ${path}/main.py 6559 &
python  ${path}/main.py 6560 &
python  ${path}/main.py 6561 &
python  ${path}/main.py 6562 &
python  ${path}/main.py 6563 &
python  ${path}/main.py 6564 &
python  ${path}/main.py 6565 &
python  ${path}/main.py 6566 &
python  ${path}/main.py 6567 &
python  ${path}/main.py 6568 &
python  ${path}/main.py 6569 &
python  ${path}/main.py 6570 &
python  ${path}/main.py 6571 &
python  ${path}/main.py 6572 &
python  ${path}/main.py 6573 &
python  ${path}/main.py 6574 &
python  ${path}/main.py 6575 &
python  ${path}/main.py 6576 &
python  ${path}/main.py 6577 &
python  ${path}/main.py 6578 &
python  ${path}/main.py 6579 &
python  ${path}/main.py 6580 &
python  ${path}/main.py 6581 &
python  ${path}/main.py 6582 &
python  ${path}/main.py 6583 &
python  ${path}/main.py 6584 &
python  ${path}/main.py 6585 &
python  ${path}/main.py 6586 &
python  ${path}/main.py 6587 &
wait
