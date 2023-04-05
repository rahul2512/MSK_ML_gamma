#!/bin/bash -l
#SBATCH -J NN_127
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9072 &
python  ${path}/main.py 9073 &
python  ${path}/main.py 9074 &
python  ${path}/main.py 9075 &
python  ${path}/main.py 9076 &
python  ${path}/main.py 9077 &
python  ${path}/main.py 9078 &
python  ${path}/main.py 9079 &
python  ${path}/main.py 9080 &
python  ${path}/main.py 9081 &
python  ${path}/main.py 9082 &
python  ${path}/main.py 9083 &
python  ${path}/main.py 9084 &
python  ${path}/main.py 9085 &
python  ${path}/main.py 9086 &
python  ${path}/main.py 9087 &
python  ${path}/main.py 9088 &
python  ${path}/main.py 9089 &
python  ${path}/main.py 9090 &
python  ${path}/main.py 9091 &
python  ${path}/main.py 9092 &
python  ${path}/main.py 9093 &
python  ${path}/main.py 9094 &
python  ${path}/main.py 9095 &
python  ${path}/main.py 9096 &
python  ${path}/main.py 9097 &
python  ${path}/main.py 9098 &
python  ${path}/main.py 9099 &
python  ${path}/main.py 9100 &
python  ${path}/main.py 9101 &
python  ${path}/main.py 9102 &
python  ${path}/main.py 9103 &
python  ${path}/main.py 9104 &
python  ${path}/main.py 9105 &
python  ${path}/main.py 9106 &
python  ${path}/main.py 9107 &
python  ${path}/main.py 9108 &
python  ${path}/main.py 9109 &
python  ${path}/main.py 9110 &
python  ${path}/main.py 9111 &
python  ${path}/main.py 9112 &
python  ${path}/main.py 9113 &
python  ${path}/main.py 9114 &
python  ${path}/main.py 9115 &
python  ${path}/main.py 9116 &
python  ${path}/main.py 9117 &
python  ${path}/main.py 9118 &
python  ${path}/main.py 9119 &
python  ${path}/main.py 9120 &
python  ${path}/main.py 9121 &
python  ${path}/main.py 9122 &
python  ${path}/main.py 9123 &
python  ${path}/main.py 9124 &
python  ${path}/main.py 9125 &
python  ${path}/main.py 9126 &
python  ${path}/main.py 9127 &
python  ${path}/main.py 9128 &
python  ${path}/main.py 9129 &
python  ${path}/main.py 9130 &
python  ${path}/main.py 9131 &
python  ${path}/main.py 9132 &
python  ${path}/main.py 9133 &
python  ${path}/main.py 9134 &
python  ${path}/main.py 9135 &
python  ${path}/main.py 9136 &
python  ${path}/main.py 9137 &
python  ${path}/main.py 9138 &
python  ${path}/main.py 9139 &
python  ${path}/main.py 9140 &
python  ${path}/main.py 9141 &
python  ${path}/main.py 9142 &
python  ${path}/main.py 9143 &
wait
