#!/bin/bash -l
#SBATCH -J NN_149
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10656 &
python  ${path}/main.py 10657 &
python  ${path}/main.py 10658 &
python  ${path}/main.py 10659 &
python  ${path}/main.py 10660 &
python  ${path}/main.py 10661 &
python  ${path}/main.py 10662 &
python  ${path}/main.py 10663 &
python  ${path}/main.py 10664 &
python  ${path}/main.py 10665 &
python  ${path}/main.py 10666 &
python  ${path}/main.py 10667 &
python  ${path}/main.py 10668 &
python  ${path}/main.py 10669 &
python  ${path}/main.py 10670 &
python  ${path}/main.py 10671 &
python  ${path}/main.py 10672 &
python  ${path}/main.py 10673 &
python  ${path}/main.py 10674 &
python  ${path}/main.py 10675 &
python  ${path}/main.py 10676 &
python  ${path}/main.py 10677 &
python  ${path}/main.py 10678 &
python  ${path}/main.py 10679 &
python  ${path}/main.py 10680 &
python  ${path}/main.py 10681 &
python  ${path}/main.py 10682 &
python  ${path}/main.py 10683 &
python  ${path}/main.py 10684 &
python  ${path}/main.py 10685 &
python  ${path}/main.py 10686 &
python  ${path}/main.py 10687 &
python  ${path}/main.py 10688 &
python  ${path}/main.py 10689 &
python  ${path}/main.py 10690 &
python  ${path}/main.py 10691 &
python  ${path}/main.py 10692 &
python  ${path}/main.py 10693 &
python  ${path}/main.py 10694 &
python  ${path}/main.py 10695 &
python  ${path}/main.py 10696 &
python  ${path}/main.py 10697 &
python  ${path}/main.py 10698 &
python  ${path}/main.py 10699 &
python  ${path}/main.py 10700 &
python  ${path}/main.py 10701 &
python  ${path}/main.py 10702 &
python  ${path}/main.py 10703 &
python  ${path}/main.py 10704 &
python  ${path}/main.py 10705 &
python  ${path}/main.py 10706 &
python  ${path}/main.py 10707 &
python  ${path}/main.py 10708 &
python  ${path}/main.py 10709 &
python  ${path}/main.py 10710 &
python  ${path}/main.py 10711 &
python  ${path}/main.py 10712 &
python  ${path}/main.py 10713 &
python  ${path}/main.py 10714 &
python  ${path}/main.py 10715 &
python  ${path}/main.py 10716 &
python  ${path}/main.py 10717 &
python  ${path}/main.py 10718 &
python  ${path}/main.py 10719 &
python  ${path}/main.py 10720 &
python  ${path}/main.py 10721 &
python  ${path}/main.py 10722 &
python  ${path}/main.py 10723 &
python  ${path}/main.py 10724 &
python  ${path}/main.py 10725 &
python  ${path}/main.py 10726 &
python  ${path}/main.py 10727 &
wait
