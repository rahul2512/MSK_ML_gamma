#!/bin/bash -l
#SBATCH -J NN_150
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10728 &
python  ${path}/main.py 10729 &
python  ${path}/main.py 10730 &
python  ${path}/main.py 10731 &
python  ${path}/main.py 10732 &
python  ${path}/main.py 10733 &
python  ${path}/main.py 10734 &
python  ${path}/main.py 10735 &
python  ${path}/main.py 10736 &
python  ${path}/main.py 10737 &
python  ${path}/main.py 10738 &
python  ${path}/main.py 10739 &
python  ${path}/main.py 10740 &
python  ${path}/main.py 10741 &
python  ${path}/main.py 10742 &
python  ${path}/main.py 10743 &
python  ${path}/main.py 10744 &
python  ${path}/main.py 10745 &
python  ${path}/main.py 10746 &
python  ${path}/main.py 10747 &
python  ${path}/main.py 10748 &
python  ${path}/main.py 10749 &
python  ${path}/main.py 10750 &
python  ${path}/main.py 10751 &
python  ${path}/main.py 10752 &
python  ${path}/main.py 10753 &
python  ${path}/main.py 10754 &
python  ${path}/main.py 10755 &
python  ${path}/main.py 10756 &
python  ${path}/main.py 10757 &
python  ${path}/main.py 10758 &
python  ${path}/main.py 10759 &
python  ${path}/main.py 10760 &
python  ${path}/main.py 10761 &
python  ${path}/main.py 10762 &
python  ${path}/main.py 10763 &
python  ${path}/main.py 10764 &
python  ${path}/main.py 10765 &
python  ${path}/main.py 10766 &
python  ${path}/main.py 10767 &
python  ${path}/main.py 10768 &
python  ${path}/main.py 10769 &
python  ${path}/main.py 10770 &
python  ${path}/main.py 10771 &
python  ${path}/main.py 10772 &
python  ${path}/main.py 10773 &
python  ${path}/main.py 10774 &
python  ${path}/main.py 10775 &
python  ${path}/main.py 10776 &
python  ${path}/main.py 10777 &
python  ${path}/main.py 10778 &
python  ${path}/main.py 10779 &
python  ${path}/main.py 10780 &
python  ${path}/main.py 10781 &
python  ${path}/main.py 10782 &
python  ${path}/main.py 10783 &
python  ${path}/main.py 10784 &
python  ${path}/main.py 10785 &
python  ${path}/main.py 10786 &
python  ${path}/main.py 10787 &
python  ${path}/main.py 10788 &
python  ${path}/main.py 10789 &
python  ${path}/main.py 10790 &
python  ${path}/main.py 10791 &
python  ${path}/main.py 10792 &
python  ${path}/main.py 10793 &
python  ${path}/main.py 10794 &
python  ${path}/main.py 10795 &
python  ${path}/main.py 10796 &
python  ${path}/main.py 10797 &
python  ${path}/main.py 10798 &
python  ${path}/main.py 10799 &
wait
