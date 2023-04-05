#!/bin/bash -l
#SBATCH -J NN_124
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8856 &
python  ${path}/main.py 8857 &
python  ${path}/main.py 8858 &
python  ${path}/main.py 8859 &
python  ${path}/main.py 8860 &
python  ${path}/main.py 8861 &
python  ${path}/main.py 8862 &
python  ${path}/main.py 8863 &
python  ${path}/main.py 8864 &
python  ${path}/main.py 8865 &
python  ${path}/main.py 8866 &
python  ${path}/main.py 8867 &
python  ${path}/main.py 8868 &
python  ${path}/main.py 8869 &
python  ${path}/main.py 8870 &
python  ${path}/main.py 8871 &
python  ${path}/main.py 8872 &
python  ${path}/main.py 8873 &
python  ${path}/main.py 8874 &
python  ${path}/main.py 8875 &
python  ${path}/main.py 8876 &
python  ${path}/main.py 8877 &
python  ${path}/main.py 8878 &
python  ${path}/main.py 8879 &
python  ${path}/main.py 8880 &
python  ${path}/main.py 8881 &
python  ${path}/main.py 8882 &
python  ${path}/main.py 8883 &
python  ${path}/main.py 8884 &
python  ${path}/main.py 8885 &
python  ${path}/main.py 8886 &
python  ${path}/main.py 8887 &
python  ${path}/main.py 8888 &
python  ${path}/main.py 8889 &
python  ${path}/main.py 8890 &
python  ${path}/main.py 8891 &
python  ${path}/main.py 8892 &
python  ${path}/main.py 8893 &
python  ${path}/main.py 8894 &
python  ${path}/main.py 8895 &
python  ${path}/main.py 8896 &
python  ${path}/main.py 8897 &
python  ${path}/main.py 8898 &
python  ${path}/main.py 8899 &
python  ${path}/main.py 8900 &
python  ${path}/main.py 8901 &
python  ${path}/main.py 8902 &
python  ${path}/main.py 8903 &
python  ${path}/main.py 8904 &
python  ${path}/main.py 8905 &
python  ${path}/main.py 8906 &
python  ${path}/main.py 8907 &
python  ${path}/main.py 8908 &
python  ${path}/main.py 8909 &
python  ${path}/main.py 8910 &
python  ${path}/main.py 8911 &
python  ${path}/main.py 8912 &
python  ${path}/main.py 8913 &
python  ${path}/main.py 8914 &
python  ${path}/main.py 8915 &
python  ${path}/main.py 8916 &
python  ${path}/main.py 8917 &
python  ${path}/main.py 8918 &
python  ${path}/main.py 8919 &
python  ${path}/main.py 8920 &
python  ${path}/main.py 8921 &
python  ${path}/main.py 8922 &
python  ${path}/main.py 8923 &
python  ${path}/main.py 8924 &
python  ${path}/main.py 8925 &
python  ${path}/main.py 8926 &
python  ${path}/main.py 8927 &
wait
