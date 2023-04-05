#!/bin/bash -l
#SBATCH -J NN_85
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6048 &
python  ${path}/main.py 6049 &
python  ${path}/main.py 6050 &
python  ${path}/main.py 6051 &
python  ${path}/main.py 6052 &
python  ${path}/main.py 6053 &
python  ${path}/main.py 6054 &
python  ${path}/main.py 6055 &
python  ${path}/main.py 6056 &
python  ${path}/main.py 6057 &
python  ${path}/main.py 6058 &
python  ${path}/main.py 6059 &
python  ${path}/main.py 6060 &
python  ${path}/main.py 6061 &
python  ${path}/main.py 6062 &
python  ${path}/main.py 6063 &
python  ${path}/main.py 6064 &
python  ${path}/main.py 6065 &
python  ${path}/main.py 6066 &
python  ${path}/main.py 6067 &
python  ${path}/main.py 6068 &
python  ${path}/main.py 6069 &
python  ${path}/main.py 6070 &
python  ${path}/main.py 6071 &
python  ${path}/main.py 6072 &
python  ${path}/main.py 6073 &
python  ${path}/main.py 6074 &
python  ${path}/main.py 6075 &
python  ${path}/main.py 6076 &
python  ${path}/main.py 6077 &
python  ${path}/main.py 6078 &
python  ${path}/main.py 6079 &
python  ${path}/main.py 6080 &
python  ${path}/main.py 6081 &
python  ${path}/main.py 6082 &
python  ${path}/main.py 6083 &
python  ${path}/main.py 6084 &
python  ${path}/main.py 6085 &
python  ${path}/main.py 6086 &
python  ${path}/main.py 6087 &
python  ${path}/main.py 6088 &
python  ${path}/main.py 6089 &
python  ${path}/main.py 6090 &
python  ${path}/main.py 6091 &
python  ${path}/main.py 6092 &
python  ${path}/main.py 6093 &
python  ${path}/main.py 6094 &
python  ${path}/main.py 6095 &
python  ${path}/main.py 6096 &
python  ${path}/main.py 6097 &
python  ${path}/main.py 6098 &
python  ${path}/main.py 6099 &
python  ${path}/main.py 6100 &
python  ${path}/main.py 6101 &
python  ${path}/main.py 6102 &
python  ${path}/main.py 6103 &
python  ${path}/main.py 6104 &
python  ${path}/main.py 6105 &
python  ${path}/main.py 6106 &
python  ${path}/main.py 6107 &
python  ${path}/main.py 6108 &
python  ${path}/main.py 6109 &
python  ${path}/main.py 6110 &
python  ${path}/main.py 6111 &
python  ${path}/main.py 6112 &
python  ${path}/main.py 6113 &
python  ${path}/main.py 6114 &
python  ${path}/main.py 6115 &
python  ${path}/main.py 6116 &
python  ${path}/main.py 6117 &
python  ${path}/main.py 6118 &
python  ${path}/main.py 6119 &
wait
