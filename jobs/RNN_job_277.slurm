#!/bin/bash -l
#SBATCH -J RNN_277
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
wait
