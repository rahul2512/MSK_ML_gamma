#!/bin/bash -l
#SBATCH -J NN_94
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6696 &
python  ${path}/main.py 6697 &
python  ${path}/main.py 6698 &
python  ${path}/main.py 6699 &
python  ${path}/main.py 6700 &
python  ${path}/main.py 6701 &
python  ${path}/main.py 6702 &
python  ${path}/main.py 6703 &
python  ${path}/main.py 6704 &
python  ${path}/main.py 6705 &
python  ${path}/main.py 6706 &
python  ${path}/main.py 6707 &
python  ${path}/main.py 6708 &
python  ${path}/main.py 6709 &
python  ${path}/main.py 6710 &
python  ${path}/main.py 6711 &
python  ${path}/main.py 6712 &
python  ${path}/main.py 6713 &
python  ${path}/main.py 6714 &
python  ${path}/main.py 6715 &
python  ${path}/main.py 6716 &
python  ${path}/main.py 6717 &
python  ${path}/main.py 6718 &
python  ${path}/main.py 6719 &
python  ${path}/main.py 6720 &
python  ${path}/main.py 6721 &
python  ${path}/main.py 6722 &
python  ${path}/main.py 6723 &
python  ${path}/main.py 6724 &
python  ${path}/main.py 6725 &
python  ${path}/main.py 6726 &
python  ${path}/main.py 6727 &
python  ${path}/main.py 6728 &
python  ${path}/main.py 6729 &
python  ${path}/main.py 6730 &
python  ${path}/main.py 6731 &
python  ${path}/main.py 6732 &
python  ${path}/main.py 6733 &
python  ${path}/main.py 6734 &
python  ${path}/main.py 6735 &
python  ${path}/main.py 6736 &
python  ${path}/main.py 6737 &
python  ${path}/main.py 6738 &
python  ${path}/main.py 6739 &
python  ${path}/main.py 6740 &
python  ${path}/main.py 6741 &
python  ${path}/main.py 6742 &
python  ${path}/main.py 6743 &
python  ${path}/main.py 6744 &
python  ${path}/main.py 6745 &
python  ${path}/main.py 6746 &
python  ${path}/main.py 6747 &
python  ${path}/main.py 6748 &
python  ${path}/main.py 6749 &
python  ${path}/main.py 6750 &
python  ${path}/main.py 6751 &
python  ${path}/main.py 6752 &
python  ${path}/main.py 6753 &
python  ${path}/main.py 6754 &
python  ${path}/main.py 6755 &
python  ${path}/main.py 6756 &
python  ${path}/main.py 6757 &
python  ${path}/main.py 6758 &
python  ${path}/main.py 6759 &
python  ${path}/main.py 6760 &
python  ${path}/main.py 6761 &
python  ${path}/main.py 6762 &
python  ${path}/main.py 6763 &
python  ${path}/main.py 6764 &
python  ${path}/main.py 6765 &
python  ${path}/main.py 6766 &
python  ${path}/main.py 6767 &
wait
