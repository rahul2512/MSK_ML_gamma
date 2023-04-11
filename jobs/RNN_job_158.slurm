#!/bin/bash -l
#SBATCH -J RNN_158
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5652 &
python  ${path}/main.py 5653 &
python  ${path}/main.py 5654 &
python  ${path}/main.py 5655 &
python  ${path}/main.py 5656 &
python  ${path}/main.py 5657 &
python  ${path}/main.py 5658 &
python  ${path}/main.py 5659 &
python  ${path}/main.py 5660 &
python  ${path}/main.py 5661 &
python  ${path}/main.py 5662 &
python  ${path}/main.py 5663 &
python  ${path}/main.py 5664 &
python  ${path}/main.py 5665 &
python  ${path}/main.py 5666 &
python  ${path}/main.py 5667 &
python  ${path}/main.py 5668 &
python  ${path}/main.py 5669 &
python  ${path}/main.py 5670 &
python  ${path}/main.py 5671 &
python  ${path}/main.py 5672 &
python  ${path}/main.py 5673 &
python  ${path}/main.py 5674 &
python  ${path}/main.py 5675 &
python  ${path}/main.py 5676 &
python  ${path}/main.py 5677 &
python  ${path}/main.py 5678 &
python  ${path}/main.py 5679 &
python  ${path}/main.py 5680 &
python  ${path}/main.py 5681 &
python  ${path}/main.py 5682 &
python  ${path}/main.py 5683 &
python  ${path}/main.py 5684 &
python  ${path}/main.py 5685 &
python  ${path}/main.py 5686 &
python  ${path}/main.py 5687 &
wait
