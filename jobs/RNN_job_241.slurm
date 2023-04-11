#!/bin/bash -l
#SBATCH -J RNN_241
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8640 &
python  ${path}/main.py 8641 &
python  ${path}/main.py 8642 &
python  ${path}/main.py 8643 &
python  ${path}/main.py 8644 &
python  ${path}/main.py 8645 &
python  ${path}/main.py 8646 &
python  ${path}/main.py 8647 &
python  ${path}/main.py 8648 &
python  ${path}/main.py 8649 &
python  ${path}/main.py 8650 &
python  ${path}/main.py 8651 &
python  ${path}/main.py 8652 &
python  ${path}/main.py 8653 &
python  ${path}/main.py 8654 &
python  ${path}/main.py 8655 &
python  ${path}/main.py 8656 &
python  ${path}/main.py 8657 &
python  ${path}/main.py 8658 &
python  ${path}/main.py 8659 &
python  ${path}/main.py 8660 &
python  ${path}/main.py 8661 &
python  ${path}/main.py 8662 &
python  ${path}/main.py 8663 &
python  ${path}/main.py 8664 &
python  ${path}/main.py 8665 &
python  ${path}/main.py 8666 &
python  ${path}/main.py 8667 &
python  ${path}/main.py 8668 &
python  ${path}/main.py 8669 &
python  ${path}/main.py 8670 &
python  ${path}/main.py 8671 &
python  ${path}/main.py 8672 &
python  ${path}/main.py 8673 &
python  ${path}/main.py 8674 &
python  ${path}/main.py 8675 &
wait
