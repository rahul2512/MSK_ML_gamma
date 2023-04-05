#!/bin/bash -l
#SBATCH -J NN_95
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6768 &
python  ${path}/main.py 6769 &
python  ${path}/main.py 6770 &
python  ${path}/main.py 6771 &
python  ${path}/main.py 6772 &
python  ${path}/main.py 6773 &
python  ${path}/main.py 6774 &
python  ${path}/main.py 6775 &
python  ${path}/main.py 6776 &
python  ${path}/main.py 6777 &
python  ${path}/main.py 6778 &
python  ${path}/main.py 6779 &
python  ${path}/main.py 6780 &
python  ${path}/main.py 6781 &
python  ${path}/main.py 6782 &
python  ${path}/main.py 6783 &
python  ${path}/main.py 6784 &
python  ${path}/main.py 6785 &
python  ${path}/main.py 6786 &
python  ${path}/main.py 6787 &
python  ${path}/main.py 6788 &
python  ${path}/main.py 6789 &
python  ${path}/main.py 6790 &
python  ${path}/main.py 6791 &
python  ${path}/main.py 6792 &
python  ${path}/main.py 6793 &
python  ${path}/main.py 6794 &
python  ${path}/main.py 6795 &
python  ${path}/main.py 6796 &
python  ${path}/main.py 6797 &
python  ${path}/main.py 6798 &
python  ${path}/main.py 6799 &
python  ${path}/main.py 6800 &
python  ${path}/main.py 6801 &
python  ${path}/main.py 6802 &
python  ${path}/main.py 6803 &
python  ${path}/main.py 6804 &
python  ${path}/main.py 6805 &
python  ${path}/main.py 6806 &
python  ${path}/main.py 6807 &
python  ${path}/main.py 6808 &
python  ${path}/main.py 6809 &
python  ${path}/main.py 6810 &
python  ${path}/main.py 6811 &
python  ${path}/main.py 6812 &
python  ${path}/main.py 6813 &
python  ${path}/main.py 6814 &
python  ${path}/main.py 6815 &
python  ${path}/main.py 6816 &
python  ${path}/main.py 6817 &
python  ${path}/main.py 6818 &
python  ${path}/main.py 6819 &
python  ${path}/main.py 6820 &
python  ${path}/main.py 6821 &
python  ${path}/main.py 6822 &
python  ${path}/main.py 6823 &
python  ${path}/main.py 6824 &
python  ${path}/main.py 6825 &
python  ${path}/main.py 6826 &
python  ${path}/main.py 6827 &
python  ${path}/main.py 6828 &
python  ${path}/main.py 6829 &
python  ${path}/main.py 6830 &
python  ${path}/main.py 6831 &
python  ${path}/main.py 6832 &
python  ${path}/main.py 6833 &
python  ${path}/main.py 6834 &
python  ${path}/main.py 6835 &
python  ${path}/main.py 6836 &
python  ${path}/main.py 6837 &
python  ${path}/main.py 6838 &
python  ${path}/main.py 6839 &
wait
