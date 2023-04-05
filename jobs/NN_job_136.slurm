#!/bin/bash -l
#SBATCH -J NN_136
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9720 &
python  ${path}/main.py 9721 &
python  ${path}/main.py 9722 &
python  ${path}/main.py 9723 &
python  ${path}/main.py 9724 &
python  ${path}/main.py 9725 &
python  ${path}/main.py 9726 &
python  ${path}/main.py 9727 &
python  ${path}/main.py 9728 &
python  ${path}/main.py 9729 &
python  ${path}/main.py 9730 &
python  ${path}/main.py 9731 &
python  ${path}/main.py 9732 &
python  ${path}/main.py 9733 &
python  ${path}/main.py 9734 &
python  ${path}/main.py 9735 &
python  ${path}/main.py 9736 &
python  ${path}/main.py 9737 &
python  ${path}/main.py 9738 &
python  ${path}/main.py 9739 &
python  ${path}/main.py 9740 &
python  ${path}/main.py 9741 &
python  ${path}/main.py 9742 &
python  ${path}/main.py 9743 &
python  ${path}/main.py 9744 &
python  ${path}/main.py 9745 &
python  ${path}/main.py 9746 &
python  ${path}/main.py 9747 &
python  ${path}/main.py 9748 &
python  ${path}/main.py 9749 &
python  ${path}/main.py 9750 &
python  ${path}/main.py 9751 &
python  ${path}/main.py 9752 &
python  ${path}/main.py 9753 &
python  ${path}/main.py 9754 &
python  ${path}/main.py 9755 &
python  ${path}/main.py 9756 &
python  ${path}/main.py 9757 &
python  ${path}/main.py 9758 &
python  ${path}/main.py 9759 &
python  ${path}/main.py 9760 &
python  ${path}/main.py 9761 &
python  ${path}/main.py 9762 &
python  ${path}/main.py 9763 &
python  ${path}/main.py 9764 &
python  ${path}/main.py 9765 &
python  ${path}/main.py 9766 &
python  ${path}/main.py 9767 &
python  ${path}/main.py 9768 &
python  ${path}/main.py 9769 &
python  ${path}/main.py 9770 &
python  ${path}/main.py 9771 &
python  ${path}/main.py 9772 &
python  ${path}/main.py 9773 &
python  ${path}/main.py 9774 &
python  ${path}/main.py 9775 &
python  ${path}/main.py 9776 &
python  ${path}/main.py 9777 &
python  ${path}/main.py 9778 &
python  ${path}/main.py 9779 &
python  ${path}/main.py 9780 &
python  ${path}/main.py 9781 &
python  ${path}/main.py 9782 &
python  ${path}/main.py 9783 &
python  ${path}/main.py 9784 &
python  ${path}/main.py 9785 &
python  ${path}/main.py 9786 &
python  ${path}/main.py 9787 &
python  ${path}/main.py 9788 &
python  ${path}/main.py 9789 &
python  ${path}/main.py 9790 &
python  ${path}/main.py 9791 &
wait
