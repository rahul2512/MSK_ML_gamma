#!/bin/bash -l
#SBATCH -J NN_139
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9936 &
python  ${path}/main.py 9937 &
python  ${path}/main.py 9938 &
python  ${path}/main.py 9939 &
python  ${path}/main.py 9940 &
python  ${path}/main.py 9941 &
python  ${path}/main.py 9942 &
python  ${path}/main.py 9943 &
python  ${path}/main.py 9944 &
python  ${path}/main.py 9945 &
python  ${path}/main.py 9946 &
python  ${path}/main.py 9947 &
python  ${path}/main.py 9948 &
python  ${path}/main.py 9949 &
python  ${path}/main.py 9950 &
python  ${path}/main.py 9951 &
python  ${path}/main.py 9952 &
python  ${path}/main.py 9953 &
python  ${path}/main.py 9954 &
python  ${path}/main.py 9955 &
python  ${path}/main.py 9956 &
python  ${path}/main.py 9957 &
python  ${path}/main.py 9958 &
python  ${path}/main.py 9959 &
python  ${path}/main.py 9960 &
python  ${path}/main.py 9961 &
python  ${path}/main.py 9962 &
python  ${path}/main.py 9963 &
python  ${path}/main.py 9964 &
python  ${path}/main.py 9965 &
python  ${path}/main.py 9966 &
python  ${path}/main.py 9967 &
python  ${path}/main.py 9968 &
python  ${path}/main.py 9969 &
python  ${path}/main.py 9970 &
python  ${path}/main.py 9971 &
python  ${path}/main.py 9972 &
python  ${path}/main.py 9973 &
python  ${path}/main.py 9974 &
python  ${path}/main.py 9975 &
python  ${path}/main.py 9976 &
python  ${path}/main.py 9977 &
python  ${path}/main.py 9978 &
python  ${path}/main.py 9979 &
python  ${path}/main.py 9980 &
python  ${path}/main.py 9981 &
python  ${path}/main.py 9982 &
python  ${path}/main.py 9983 &
python  ${path}/main.py 9984 &
python  ${path}/main.py 9985 &
python  ${path}/main.py 9986 &
python  ${path}/main.py 9987 &
python  ${path}/main.py 9988 &
python  ${path}/main.py 9989 &
python  ${path}/main.py 9990 &
python  ${path}/main.py 9991 &
python  ${path}/main.py 9992 &
python  ${path}/main.py 9993 &
python  ${path}/main.py 9994 &
python  ${path}/main.py 9995 &
python  ${path}/main.py 9996 &
python  ${path}/main.py 9997 &
python  ${path}/main.py 9998 &
python  ${path}/main.py 9999 &
python  ${path}/main.py 10000 &
python  ${path}/main.py 10001 &
python  ${path}/main.py 10002 &
python  ${path}/main.py 10003 &
python  ${path}/main.py 10004 &
python  ${path}/main.py 10005 &
python  ${path}/main.py 10006 &
python  ${path}/main.py 10007 &
wait
