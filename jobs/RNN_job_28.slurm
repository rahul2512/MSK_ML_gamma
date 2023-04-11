#!/bin/bash -l
#SBATCH -J RNN_28
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 972 &
python  ${path}/main.py 973 &
python  ${path}/main.py 974 &
python  ${path}/main.py 975 &
python  ${path}/main.py 976 &
python  ${path}/main.py 977 &
python  ${path}/main.py 978 &
python  ${path}/main.py 979 &
python  ${path}/main.py 980 &
python  ${path}/main.py 981 &
python  ${path}/main.py 982 &
python  ${path}/main.py 983 &
python  ${path}/main.py 984 &
python  ${path}/main.py 985 &
python  ${path}/main.py 986 &
python  ${path}/main.py 987 &
python  ${path}/main.py 988 &
python  ${path}/main.py 989 &
python  ${path}/main.py 990 &
python  ${path}/main.py 991 &
python  ${path}/main.py 992 &
python  ${path}/main.py 993 &
python  ${path}/main.py 994 &
python  ${path}/main.py 995 &
python  ${path}/main.py 996 &
python  ${path}/main.py 997 &
python  ${path}/main.py 998 &
python  ${path}/main.py 999 &
python  ${path}/main.py 1000 &
python  ${path}/main.py 1001 &
python  ${path}/main.py 1002 &
python  ${path}/main.py 1003 &
python  ${path}/main.py 1004 &
python  ${path}/main.py 1005 &
python  ${path}/main.py 1006 &
python  ${path}/main.py 1007 &
wait
