#!/bin/bash -l
#SBATCH -J NN_70
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
python  ${path}/main.py 5004 &
python  ${path}/main.py 5005 &
python  ${path}/main.py 5006 &
python  ${path}/main.py 5007 &
python  ${path}/main.py 5008 &
python  ${path}/main.py 5009 &
python  ${path}/main.py 5010 &
python  ${path}/main.py 5011 &
python  ${path}/main.py 5012 &
python  ${path}/main.py 5013 &
python  ${path}/main.py 5014 &
python  ${path}/main.py 5015 &
python  ${path}/main.py 5016 &
python  ${path}/main.py 5017 &
python  ${path}/main.py 5018 &
python  ${path}/main.py 5019 &
python  ${path}/main.py 5020 &
python  ${path}/main.py 5021 &
python  ${path}/main.py 5022 &
python  ${path}/main.py 5023 &
python  ${path}/main.py 5024 &
python  ${path}/main.py 5025 &
python  ${path}/main.py 5026 &
python  ${path}/main.py 5027 &
python  ${path}/main.py 5028 &
python  ${path}/main.py 5029 &
python  ${path}/main.py 5030 &
python  ${path}/main.py 5031 &
python  ${path}/main.py 5032 &
python  ${path}/main.py 5033 &
python  ${path}/main.py 5034 &
python  ${path}/main.py 5035 &
python  ${path}/main.py 5036 &
python  ${path}/main.py 5037 &
python  ${path}/main.py 5038 &
python  ${path}/main.py 5039 &
wait
