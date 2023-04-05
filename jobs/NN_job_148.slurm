#!/bin/bash -l
#SBATCH -J NN_148
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10584 &
python  ${path}/main.py 10585 &
python  ${path}/main.py 10586 &
python  ${path}/main.py 10587 &
python  ${path}/main.py 10588 &
python  ${path}/main.py 10589 &
python  ${path}/main.py 10590 &
python  ${path}/main.py 10591 &
python  ${path}/main.py 10592 &
python  ${path}/main.py 10593 &
python  ${path}/main.py 10594 &
python  ${path}/main.py 10595 &
python  ${path}/main.py 10596 &
python  ${path}/main.py 10597 &
python  ${path}/main.py 10598 &
python  ${path}/main.py 10599 &
python  ${path}/main.py 10600 &
python  ${path}/main.py 10601 &
python  ${path}/main.py 10602 &
python  ${path}/main.py 10603 &
python  ${path}/main.py 10604 &
python  ${path}/main.py 10605 &
python  ${path}/main.py 10606 &
python  ${path}/main.py 10607 &
python  ${path}/main.py 10608 &
python  ${path}/main.py 10609 &
python  ${path}/main.py 10610 &
python  ${path}/main.py 10611 &
python  ${path}/main.py 10612 &
python  ${path}/main.py 10613 &
python  ${path}/main.py 10614 &
python  ${path}/main.py 10615 &
python  ${path}/main.py 10616 &
python  ${path}/main.py 10617 &
python  ${path}/main.py 10618 &
python  ${path}/main.py 10619 &
python  ${path}/main.py 10620 &
python  ${path}/main.py 10621 &
python  ${path}/main.py 10622 &
python  ${path}/main.py 10623 &
python  ${path}/main.py 10624 &
python  ${path}/main.py 10625 &
python  ${path}/main.py 10626 &
python  ${path}/main.py 10627 &
python  ${path}/main.py 10628 &
python  ${path}/main.py 10629 &
python  ${path}/main.py 10630 &
python  ${path}/main.py 10631 &
python  ${path}/main.py 10632 &
python  ${path}/main.py 10633 &
python  ${path}/main.py 10634 &
python  ${path}/main.py 10635 &
python  ${path}/main.py 10636 &
python  ${path}/main.py 10637 &
python  ${path}/main.py 10638 &
python  ${path}/main.py 10639 &
python  ${path}/main.py 10640 &
python  ${path}/main.py 10641 &
python  ${path}/main.py 10642 &
python  ${path}/main.py 10643 &
python  ${path}/main.py 10644 &
python  ${path}/main.py 10645 &
python  ${path}/main.py 10646 &
python  ${path}/main.py 10647 &
python  ${path}/main.py 10648 &
python  ${path}/main.py 10649 &
python  ${path}/main.py 10650 &
python  ${path}/main.py 10651 &
python  ${path}/main.py 10652 &
python  ${path}/main.py 10653 &
python  ${path}/main.py 10654 &
python  ${path}/main.py 10655 &
wait
