#!/bin/bash -l
#SBATCH -J NN_11
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 720 &
python  ${path}/main.py 721 &
python  ${path}/main.py 722 &
python  ${path}/main.py 723 &
python  ${path}/main.py 724 &
python  ${path}/main.py 725 &
python  ${path}/main.py 726 &
python  ${path}/main.py 727 &
python  ${path}/main.py 728 &
python  ${path}/main.py 729 &
python  ${path}/main.py 730 &
python  ${path}/main.py 731 &
python  ${path}/main.py 732 &
python  ${path}/main.py 733 &
python  ${path}/main.py 734 &
python  ${path}/main.py 735 &
python  ${path}/main.py 736 &
python  ${path}/main.py 737 &
python  ${path}/main.py 738 &
python  ${path}/main.py 739 &
python  ${path}/main.py 740 &
python  ${path}/main.py 741 &
python  ${path}/main.py 742 &
python  ${path}/main.py 743 &
python  ${path}/main.py 744 &
python  ${path}/main.py 745 &
python  ${path}/main.py 746 &
python  ${path}/main.py 747 &
python  ${path}/main.py 748 &
python  ${path}/main.py 749 &
python  ${path}/main.py 750 &
python  ${path}/main.py 751 &
python  ${path}/main.py 752 &
python  ${path}/main.py 753 &
python  ${path}/main.py 754 &
python  ${path}/main.py 755 &
python  ${path}/main.py 756 &
python  ${path}/main.py 757 &
python  ${path}/main.py 758 &
python  ${path}/main.py 759 &
python  ${path}/main.py 760 &
python  ${path}/main.py 761 &
python  ${path}/main.py 762 &
python  ${path}/main.py 763 &
python  ${path}/main.py 764 &
python  ${path}/main.py 765 &
python  ${path}/main.py 766 &
python  ${path}/main.py 767 &
python  ${path}/main.py 768 &
python  ${path}/main.py 769 &
python  ${path}/main.py 770 &
python  ${path}/main.py 771 &
python  ${path}/main.py 772 &
python  ${path}/main.py 773 &
python  ${path}/main.py 774 &
python  ${path}/main.py 775 &
python  ${path}/main.py 776 &
python  ${path}/main.py 777 &
python  ${path}/main.py 778 &
python  ${path}/main.py 779 &
python  ${path}/main.py 780 &
python  ${path}/main.py 781 &
python  ${path}/main.py 782 &
python  ${path}/main.py 783 &
python  ${path}/main.py 784 &
python  ${path}/main.py 785 &
python  ${path}/main.py 786 &
python  ${path}/main.py 787 &
python  ${path}/main.py 788 &
python  ${path}/main.py 789 &
python  ${path}/main.py 790 &
python  ${path}/main.py 791 &
wait
