#!/bin/bash -l
#SBATCH -J NN_123
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8784 &
python  ${path}/main.py 8785 &
python  ${path}/main.py 8786 &
python  ${path}/main.py 8787 &
python  ${path}/main.py 8788 &
python  ${path}/main.py 8789 &
python  ${path}/main.py 8790 &
python  ${path}/main.py 8791 &
python  ${path}/main.py 8792 &
python  ${path}/main.py 8793 &
python  ${path}/main.py 8794 &
python  ${path}/main.py 8795 &
python  ${path}/main.py 8796 &
python  ${path}/main.py 8797 &
python  ${path}/main.py 8798 &
python  ${path}/main.py 8799 &
python  ${path}/main.py 8800 &
python  ${path}/main.py 8801 &
python  ${path}/main.py 8802 &
python  ${path}/main.py 8803 &
python  ${path}/main.py 8804 &
python  ${path}/main.py 8805 &
python  ${path}/main.py 8806 &
python  ${path}/main.py 8807 &
python  ${path}/main.py 8808 &
python  ${path}/main.py 8809 &
python  ${path}/main.py 8810 &
python  ${path}/main.py 8811 &
python  ${path}/main.py 8812 &
python  ${path}/main.py 8813 &
python  ${path}/main.py 8814 &
python  ${path}/main.py 8815 &
python  ${path}/main.py 8816 &
python  ${path}/main.py 8817 &
python  ${path}/main.py 8818 &
python  ${path}/main.py 8819 &
python  ${path}/main.py 8820 &
python  ${path}/main.py 8821 &
python  ${path}/main.py 8822 &
python  ${path}/main.py 8823 &
python  ${path}/main.py 8824 &
python  ${path}/main.py 8825 &
python  ${path}/main.py 8826 &
python  ${path}/main.py 8827 &
python  ${path}/main.py 8828 &
python  ${path}/main.py 8829 &
python  ${path}/main.py 8830 &
python  ${path}/main.py 8831 &
python  ${path}/main.py 8832 &
python  ${path}/main.py 8833 &
python  ${path}/main.py 8834 &
python  ${path}/main.py 8835 &
python  ${path}/main.py 8836 &
python  ${path}/main.py 8837 &
python  ${path}/main.py 8838 &
python  ${path}/main.py 8839 &
python  ${path}/main.py 8840 &
python  ${path}/main.py 8841 &
python  ${path}/main.py 8842 &
python  ${path}/main.py 8843 &
python  ${path}/main.py 8844 &
python  ${path}/main.py 8845 &
python  ${path}/main.py 8846 &
python  ${path}/main.py 8847 &
python  ${path}/main.py 8848 &
python  ${path}/main.py 8849 &
python  ${path}/main.py 8850 &
python  ${path}/main.py 8851 &
python  ${path}/main.py 8852 &
python  ${path}/main.py 8853 &
python  ${path}/main.py 8854 &
python  ${path}/main.py 8855 &
wait
