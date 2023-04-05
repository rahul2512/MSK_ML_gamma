#!/bin/bash -l
#SBATCH -J NN_106
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
python  ${path}/main.py 7596 &
python  ${path}/main.py 7597 &
python  ${path}/main.py 7598 &
python  ${path}/main.py 7599 &
python  ${path}/main.py 7600 &
python  ${path}/main.py 7601 &
python  ${path}/main.py 7602 &
python  ${path}/main.py 7603 &
python  ${path}/main.py 7604 &
python  ${path}/main.py 7605 &
python  ${path}/main.py 7606 &
python  ${path}/main.py 7607 &
python  ${path}/main.py 7608 &
python  ${path}/main.py 7609 &
python  ${path}/main.py 7610 &
python  ${path}/main.py 7611 &
python  ${path}/main.py 7612 &
python  ${path}/main.py 7613 &
python  ${path}/main.py 7614 &
python  ${path}/main.py 7615 &
python  ${path}/main.py 7616 &
python  ${path}/main.py 7617 &
python  ${path}/main.py 7618 &
python  ${path}/main.py 7619 &
python  ${path}/main.py 7620 &
python  ${path}/main.py 7621 &
python  ${path}/main.py 7622 &
python  ${path}/main.py 7623 &
python  ${path}/main.py 7624 &
python  ${path}/main.py 7625 &
python  ${path}/main.py 7626 &
python  ${path}/main.py 7627 &
python  ${path}/main.py 7628 &
python  ${path}/main.py 7629 &
python  ${path}/main.py 7630 &
python  ${path}/main.py 7631 &
wait
