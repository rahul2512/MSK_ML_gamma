#!/bin/bash -l
#SBATCH -J RNN_192
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6876 &
python  ${path}/main.py 6877 &
python  ${path}/main.py 6878 &
python  ${path}/main.py 6879 &
python  ${path}/main.py 6880 &
python  ${path}/main.py 6881 &
python  ${path}/main.py 6882 &
python  ${path}/main.py 6883 &
python  ${path}/main.py 6884 &
python  ${path}/main.py 6885 &
python  ${path}/main.py 6886 &
python  ${path}/main.py 6887 &
python  ${path}/main.py 6888 &
python  ${path}/main.py 6889 &
python  ${path}/main.py 6890 &
python  ${path}/main.py 6891 &
python  ${path}/main.py 6892 &
python  ${path}/main.py 6893 &
python  ${path}/main.py 6894 &
python  ${path}/main.py 6895 &
python  ${path}/main.py 6896 &
python  ${path}/main.py 6897 &
python  ${path}/main.py 6898 &
python  ${path}/main.py 6899 &
python  ${path}/main.py 6900 &
python  ${path}/main.py 6901 &
python  ${path}/main.py 6902 &
python  ${path}/main.py 6903 &
python  ${path}/main.py 6904 &
python  ${path}/main.py 6905 &
python  ${path}/main.py 6906 &
python  ${path}/main.py 6907 &
python  ${path}/main.py 6908 &
python  ${path}/main.py 6909 &
python  ${path}/main.py 6910 &
python  ${path}/main.py 6911 &
wait
