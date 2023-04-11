#!/bin/bash -l
#SBATCH -J RNN_211
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7560 &
python  ${path}/main.py 7561 &
python  ${path}/main.py 7562 &
python  ${path}/main.py 7563 &
python  ${path}/main.py 7564 &
python  ${path}/main.py 7565 &
python  ${path}/main.py 7566 &
python  ${path}/main.py 7567 &
python  ${path}/main.py 7568 &
python  ${path}/main.py 7569 &
python  ${path}/main.py 7570 &
python  ${path}/main.py 7571 &
python  ${path}/main.py 7572 &
python  ${path}/main.py 7573 &
python  ${path}/main.py 7574 &
python  ${path}/main.py 7575 &
python  ${path}/main.py 7576 &
python  ${path}/main.py 7577 &
python  ${path}/main.py 7578 &
python  ${path}/main.py 7579 &
python  ${path}/main.py 7580 &
python  ${path}/main.py 7581 &
python  ${path}/main.py 7582 &
python  ${path}/main.py 7583 &
python  ${path}/main.py 7584 &
python  ${path}/main.py 7585 &
python  ${path}/main.py 7586 &
python  ${path}/main.py 7587 &
python  ${path}/main.py 7588 &
python  ${path}/main.py 7589 &
python  ${path}/main.py 7590 &
python  ${path}/main.py 7591 &
python  ${path}/main.py 7592 &
python  ${path}/main.py 7593 &
python  ${path}/main.py 7594 &
python  ${path}/main.py 7595 &
wait
