#!/bin/bash -l
#SBATCH -J RNN_85
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3024 &
python  ${path}/main.py 3025 &
python  ${path}/main.py 3026 &
python  ${path}/main.py 3027 &
python  ${path}/main.py 3028 &
python  ${path}/main.py 3029 &
python  ${path}/main.py 3030 &
python  ${path}/main.py 3031 &
python  ${path}/main.py 3032 &
python  ${path}/main.py 3033 &
python  ${path}/main.py 3034 &
python  ${path}/main.py 3035 &
python  ${path}/main.py 3036 &
python  ${path}/main.py 3037 &
python  ${path}/main.py 3038 &
python  ${path}/main.py 3039 &
python  ${path}/main.py 3040 &
python  ${path}/main.py 3041 &
python  ${path}/main.py 3042 &
python  ${path}/main.py 3043 &
python  ${path}/main.py 3044 &
python  ${path}/main.py 3045 &
python  ${path}/main.py 3046 &
python  ${path}/main.py 3047 &
python  ${path}/main.py 3048 &
python  ${path}/main.py 3049 &
python  ${path}/main.py 3050 &
python  ${path}/main.py 3051 &
python  ${path}/main.py 3052 &
python  ${path}/main.py 3053 &
python  ${path}/main.py 3054 &
python  ${path}/main.py 3055 &
python  ${path}/main.py 3056 &
python  ${path}/main.py 3057 &
python  ${path}/main.py 3058 &
python  ${path}/main.py 3059 &
wait
