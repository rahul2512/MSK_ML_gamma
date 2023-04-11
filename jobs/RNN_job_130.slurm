#!/bin/bash -l
#SBATCH -J RNN_130
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4644 &
python  ${path}/main.py 4645 &
python  ${path}/main.py 4646 &
python  ${path}/main.py 4647 &
python  ${path}/main.py 4648 &
python  ${path}/main.py 4649 &
python  ${path}/main.py 4650 &
python  ${path}/main.py 4651 &
python  ${path}/main.py 4652 &
python  ${path}/main.py 4653 &
python  ${path}/main.py 4654 &
python  ${path}/main.py 4655 &
python  ${path}/main.py 4656 &
python  ${path}/main.py 4657 &
python  ${path}/main.py 4658 &
python  ${path}/main.py 4659 &
python  ${path}/main.py 4660 &
python  ${path}/main.py 4661 &
python  ${path}/main.py 4662 &
python  ${path}/main.py 4663 &
python  ${path}/main.py 4664 &
python  ${path}/main.py 4665 &
python  ${path}/main.py 4666 &
python  ${path}/main.py 4667 &
python  ${path}/main.py 4668 &
python  ${path}/main.py 4669 &
python  ${path}/main.py 4670 &
python  ${path}/main.py 4671 &
python  ${path}/main.py 4672 &
python  ${path}/main.py 4673 &
python  ${path}/main.py 4674 &
python  ${path}/main.py 4675 &
python  ${path}/main.py 4676 &
python  ${path}/main.py 4677 &
python  ${path}/main.py 4678 &
python  ${path}/main.py 4679 &
wait
