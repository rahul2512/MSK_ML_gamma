#!/bin/bash -l
#SBATCH -J RNN_166
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5940 &
python  ${path}/main.py 5941 &
python  ${path}/main.py 5942 &
python  ${path}/main.py 5943 &
python  ${path}/main.py 5944 &
python  ${path}/main.py 5945 &
python  ${path}/main.py 5946 &
python  ${path}/main.py 5947 &
python  ${path}/main.py 5948 &
python  ${path}/main.py 5949 &
python  ${path}/main.py 5950 &
python  ${path}/main.py 5951 &
python  ${path}/main.py 5952 &
python  ${path}/main.py 5953 &
python  ${path}/main.py 5954 &
python  ${path}/main.py 5955 &
python  ${path}/main.py 5956 &
python  ${path}/main.py 5957 &
python  ${path}/main.py 5958 &
python  ${path}/main.py 5959 &
python  ${path}/main.py 5960 &
python  ${path}/main.py 5961 &
python  ${path}/main.py 5962 &
python  ${path}/main.py 5963 &
python  ${path}/main.py 5964 &
python  ${path}/main.py 5965 &
python  ${path}/main.py 5966 &
python  ${path}/main.py 5967 &
python  ${path}/main.py 5968 &
python  ${path}/main.py 5969 &
python  ${path}/main.py 5970 &
python  ${path}/main.py 5971 &
python  ${path}/main.py 5972 &
python  ${path}/main.py 5973 &
python  ${path}/main.py 5974 &
python  ${path}/main.py 5975 &
wait
