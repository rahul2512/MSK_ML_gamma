#!/bin/bash -l
#SBATCH -J RNN_139
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4968 &
python  ${path}/main.py 4969 &
python  ${path}/main.py 4970 &
python  ${path}/main.py 4971 &
python  ${path}/main.py 4972 &
python  ${path}/main.py 4973 &
python  ${path}/main.py 4974 &
python  ${path}/main.py 4975 &
python  ${path}/main.py 4976 &
python  ${path}/main.py 4977 &
python  ${path}/main.py 4978 &
python  ${path}/main.py 4979 &
python  ${path}/main.py 4980 &
python  ${path}/main.py 4981 &
python  ${path}/main.py 4982 &
python  ${path}/main.py 4983 &
python  ${path}/main.py 4984 &
python  ${path}/main.py 4985 &
python  ${path}/main.py 4986 &
python  ${path}/main.py 4987 &
python  ${path}/main.py 4988 &
python  ${path}/main.py 4989 &
python  ${path}/main.py 4990 &
python  ${path}/main.py 4991 &
python  ${path}/main.py 4992 &
python  ${path}/main.py 4993 &
python  ${path}/main.py 4994 &
python  ${path}/main.py 4995 &
python  ${path}/main.py 4996 &
python  ${path}/main.py 4997 &
python  ${path}/main.py 4998 &
python  ${path}/main.py 4999 &
python  ${path}/main.py 5000 &
python  ${path}/main.py 5001 &
python  ${path}/main.py 5002 &
python  ${path}/main.py 5003 &
wait
