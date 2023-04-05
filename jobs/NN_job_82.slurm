#!/bin/bash -l
#SBATCH -J NN_82
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5832 &
python  ${path}/main.py 5833 &
python  ${path}/main.py 5834 &
python  ${path}/main.py 5835 &
python  ${path}/main.py 5836 &
python  ${path}/main.py 5837 &
python  ${path}/main.py 5838 &
python  ${path}/main.py 5839 &
python  ${path}/main.py 5840 &
python  ${path}/main.py 5841 &
python  ${path}/main.py 5842 &
python  ${path}/main.py 5843 &
python  ${path}/main.py 5844 &
python  ${path}/main.py 5845 &
python  ${path}/main.py 5846 &
python  ${path}/main.py 5847 &
python  ${path}/main.py 5848 &
python  ${path}/main.py 5849 &
python  ${path}/main.py 5850 &
python  ${path}/main.py 5851 &
python  ${path}/main.py 5852 &
python  ${path}/main.py 5853 &
python  ${path}/main.py 5854 &
python  ${path}/main.py 5855 &
python  ${path}/main.py 5856 &
python  ${path}/main.py 5857 &
python  ${path}/main.py 5858 &
python  ${path}/main.py 5859 &
python  ${path}/main.py 5860 &
python  ${path}/main.py 5861 &
python  ${path}/main.py 5862 &
python  ${path}/main.py 5863 &
python  ${path}/main.py 5864 &
python  ${path}/main.py 5865 &
python  ${path}/main.py 5866 &
python  ${path}/main.py 5867 &
python  ${path}/main.py 5868 &
python  ${path}/main.py 5869 &
python  ${path}/main.py 5870 &
python  ${path}/main.py 5871 &
python  ${path}/main.py 5872 &
python  ${path}/main.py 5873 &
python  ${path}/main.py 5874 &
python  ${path}/main.py 5875 &
python  ${path}/main.py 5876 &
python  ${path}/main.py 5877 &
python  ${path}/main.py 5878 &
python  ${path}/main.py 5879 &
python  ${path}/main.py 5880 &
python  ${path}/main.py 5881 &
python  ${path}/main.py 5882 &
python  ${path}/main.py 5883 &
python  ${path}/main.py 5884 &
python  ${path}/main.py 5885 &
python  ${path}/main.py 5886 &
python  ${path}/main.py 5887 &
python  ${path}/main.py 5888 &
python  ${path}/main.py 5889 &
python  ${path}/main.py 5890 &
python  ${path}/main.py 5891 &
python  ${path}/main.py 5892 &
python  ${path}/main.py 5893 &
python  ${path}/main.py 5894 &
python  ${path}/main.py 5895 &
python  ${path}/main.py 5896 &
python  ${path}/main.py 5897 &
python  ${path}/main.py 5898 &
python  ${path}/main.py 5899 &
python  ${path}/main.py 5900 &
python  ${path}/main.py 5901 &
python  ${path}/main.py 5902 &
python  ${path}/main.py 5903 &
wait
