#!/bin/bash -l
#SBATCH -J NN_154
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11016 &
python  ${path}/main.py 11017 &
python  ${path}/main.py 11018 &
python  ${path}/main.py 11019 &
python  ${path}/main.py 11020 &
python  ${path}/main.py 11021 &
python  ${path}/main.py 11022 &
python  ${path}/main.py 11023 &
python  ${path}/main.py 11024 &
python  ${path}/main.py 11025 &
python  ${path}/main.py 11026 &
python  ${path}/main.py 11027 &
python  ${path}/main.py 11028 &
python  ${path}/main.py 11029 &
python  ${path}/main.py 11030 &
python  ${path}/main.py 11031 &
python  ${path}/main.py 11032 &
python  ${path}/main.py 11033 &
python  ${path}/main.py 11034 &
python  ${path}/main.py 11035 &
python  ${path}/main.py 11036 &
python  ${path}/main.py 11037 &
python  ${path}/main.py 11038 &
python  ${path}/main.py 11039 &
python  ${path}/main.py 11040 &
python  ${path}/main.py 11041 &
python  ${path}/main.py 11042 &
python  ${path}/main.py 11043 &
python  ${path}/main.py 11044 &
python  ${path}/main.py 11045 &
python  ${path}/main.py 11046 &
python  ${path}/main.py 11047 &
python  ${path}/main.py 11048 &
python  ${path}/main.py 11049 &
python  ${path}/main.py 11050 &
python  ${path}/main.py 11051 &
python  ${path}/main.py 11052 &
python  ${path}/main.py 11053 &
python  ${path}/main.py 11054 &
python  ${path}/main.py 11055 &
python  ${path}/main.py 11056 &
python  ${path}/main.py 11057 &
python  ${path}/main.py 11058 &
python  ${path}/main.py 11059 &
python  ${path}/main.py 11060 &
python  ${path}/main.py 11061 &
python  ${path}/main.py 11062 &
python  ${path}/main.py 11063 &
python  ${path}/main.py 11064 &
python  ${path}/main.py 11065 &
python  ${path}/main.py 11066 &
python  ${path}/main.py 11067 &
python  ${path}/main.py 11068 &
python  ${path}/main.py 11069 &
python  ${path}/main.py 11070 &
python  ${path}/main.py 11071 &
python  ${path}/main.py 11072 &
python  ${path}/main.py 11073 &
python  ${path}/main.py 11074 &
python  ${path}/main.py 11075 &
python  ${path}/main.py 11076 &
python  ${path}/main.py 11077 &
python  ${path}/main.py 11078 &
python  ${path}/main.py 11079 &
python  ${path}/main.py 11080 &
python  ${path}/main.py 11081 &
python  ${path}/main.py 11082 &
python  ${path}/main.py 11083 &
python  ${path}/main.py 11084 &
python  ${path}/main.py 11085 &
python  ${path}/main.py 11086 &
python  ${path}/main.py 11087 &
wait
