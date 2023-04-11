#!/bin/bash -l
#SBATCH -J RNN_269
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9648 &
python  ${path}/main.py 9649 &
python  ${path}/main.py 9650 &
python  ${path}/main.py 9651 &
python  ${path}/main.py 9652 &
python  ${path}/main.py 9653 &
python  ${path}/main.py 9654 &
python  ${path}/main.py 9655 &
python  ${path}/main.py 9656 &
python  ${path}/main.py 9657 &
python  ${path}/main.py 9658 &
python  ${path}/main.py 9659 &
python  ${path}/main.py 9660 &
python  ${path}/main.py 9661 &
python  ${path}/main.py 9662 &
python  ${path}/main.py 9663 &
python  ${path}/main.py 9664 &
python  ${path}/main.py 9665 &
python  ${path}/main.py 9666 &
python  ${path}/main.py 9667 &
python  ${path}/main.py 9668 &
python  ${path}/main.py 9669 &
python  ${path}/main.py 9670 &
python  ${path}/main.py 9671 &
python  ${path}/main.py 9672 &
python  ${path}/main.py 9673 &
python  ${path}/main.py 9674 &
python  ${path}/main.py 9675 &
python  ${path}/main.py 9676 &
python  ${path}/main.py 9677 &
python  ${path}/main.py 9678 &
python  ${path}/main.py 9679 &
python  ${path}/main.py 9680 &
python  ${path}/main.py 9681 &
python  ${path}/main.py 9682 &
python  ${path}/main.py 9683 &
wait
