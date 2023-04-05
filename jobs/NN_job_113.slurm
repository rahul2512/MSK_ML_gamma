#!/bin/bash -l
#SBATCH -J NN_113
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8064 &
python  ${path}/main.py 8065 &
python  ${path}/main.py 8066 &
python  ${path}/main.py 8067 &
python  ${path}/main.py 8068 &
python  ${path}/main.py 8069 &
python  ${path}/main.py 8070 &
python  ${path}/main.py 8071 &
python  ${path}/main.py 8072 &
python  ${path}/main.py 8073 &
python  ${path}/main.py 8074 &
python  ${path}/main.py 8075 &
python  ${path}/main.py 8076 &
python  ${path}/main.py 8077 &
python  ${path}/main.py 8078 &
python  ${path}/main.py 8079 &
python  ${path}/main.py 8080 &
python  ${path}/main.py 8081 &
python  ${path}/main.py 8082 &
python  ${path}/main.py 8083 &
python  ${path}/main.py 8084 &
python  ${path}/main.py 8085 &
python  ${path}/main.py 8086 &
python  ${path}/main.py 8087 &
python  ${path}/main.py 8088 &
python  ${path}/main.py 8089 &
python  ${path}/main.py 8090 &
python  ${path}/main.py 8091 &
python  ${path}/main.py 8092 &
python  ${path}/main.py 8093 &
python  ${path}/main.py 8094 &
python  ${path}/main.py 8095 &
python  ${path}/main.py 8096 &
python  ${path}/main.py 8097 &
python  ${path}/main.py 8098 &
python  ${path}/main.py 8099 &
python  ${path}/main.py 8100 &
python  ${path}/main.py 8101 &
python  ${path}/main.py 8102 &
python  ${path}/main.py 8103 &
python  ${path}/main.py 8104 &
python  ${path}/main.py 8105 &
python  ${path}/main.py 8106 &
python  ${path}/main.py 8107 &
python  ${path}/main.py 8108 &
python  ${path}/main.py 8109 &
python  ${path}/main.py 8110 &
python  ${path}/main.py 8111 &
python  ${path}/main.py 8112 &
python  ${path}/main.py 8113 &
python  ${path}/main.py 8114 &
python  ${path}/main.py 8115 &
python  ${path}/main.py 8116 &
python  ${path}/main.py 8117 &
python  ${path}/main.py 8118 &
python  ${path}/main.py 8119 &
python  ${path}/main.py 8120 &
python  ${path}/main.py 8121 &
python  ${path}/main.py 8122 &
python  ${path}/main.py 8123 &
python  ${path}/main.py 8124 &
python  ${path}/main.py 8125 &
python  ${path}/main.py 8126 &
python  ${path}/main.py 8127 &
python  ${path}/main.py 8128 &
python  ${path}/main.py 8129 &
python  ${path}/main.py 8130 &
python  ${path}/main.py 8131 &
python  ${path}/main.py 8132 &
python  ${path}/main.py 8133 &
python  ${path}/main.py 8134 &
python  ${path}/main.py 8135 &
wait
