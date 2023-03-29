#!/bin/bash -l
#SBATCH -J CNNLSTM_14
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 936 &
python  ${path}/main.py 937 &
python  ${path}/main.py 938 &
python  ${path}/main.py 939 &
python  ${path}/main.py 940 &
python  ${path}/main.py 941 &
python  ${path}/main.py 942 &
python  ${path}/main.py 943 &
python  ${path}/main.py 944 &
python  ${path}/main.py 945 &
python  ${path}/main.py 946 &
python  ${path}/main.py 947 &
python  ${path}/main.py 948 &
python  ${path}/main.py 949 &
python  ${path}/main.py 950 &
python  ${path}/main.py 951 &
python  ${path}/main.py 952 &
python  ${path}/main.py 953 &
python  ${path}/main.py 954 &
python  ${path}/main.py 955 &
python  ${path}/main.py 956 &
python  ${path}/main.py 957 &
python  ${path}/main.py 958 &
python  ${path}/main.py 959 &
python  ${path}/main.py 960 &
python  ${path}/main.py 961 &
python  ${path}/main.py 962 &
python  ${path}/main.py 963 &
python  ${path}/main.py 964 &
python  ${path}/main.py 965 &
python  ${path}/main.py 966 &
python  ${path}/main.py 967 &
python  ${path}/main.py 968 &
python  ${path}/main.py 969 &
python  ${path}/main.py 970 &
python  ${path}/main.py 971 &
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
