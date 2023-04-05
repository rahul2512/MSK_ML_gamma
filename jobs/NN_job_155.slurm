#!/bin/bash -l
#SBATCH -J NN_155
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11088 &
python  ${path}/main.py 11089 &
python  ${path}/main.py 11090 &
python  ${path}/main.py 11091 &
python  ${path}/main.py 11092 &
python  ${path}/main.py 11093 &
python  ${path}/main.py 11094 &
python  ${path}/main.py 11095 &
python  ${path}/main.py 11096 &
python  ${path}/main.py 11097 &
python  ${path}/main.py 11098 &
python  ${path}/main.py 11099 &
python  ${path}/main.py 11100 &
python  ${path}/main.py 11101 &
python  ${path}/main.py 11102 &
python  ${path}/main.py 11103 &
python  ${path}/main.py 11104 &
python  ${path}/main.py 11105 &
python  ${path}/main.py 11106 &
python  ${path}/main.py 11107 &
python  ${path}/main.py 11108 &
python  ${path}/main.py 11109 &
python  ${path}/main.py 11110 &
python  ${path}/main.py 11111 &
python  ${path}/main.py 11112 &
python  ${path}/main.py 11113 &
python  ${path}/main.py 11114 &
python  ${path}/main.py 11115 &
python  ${path}/main.py 11116 &
python  ${path}/main.py 11117 &
python  ${path}/main.py 11118 &
python  ${path}/main.py 11119 &
python  ${path}/main.py 11120 &
python  ${path}/main.py 11121 &
python  ${path}/main.py 11122 &
python  ${path}/main.py 11123 &
python  ${path}/main.py 11124 &
python  ${path}/main.py 11125 &
python  ${path}/main.py 11126 &
python  ${path}/main.py 11127 &
python  ${path}/main.py 11128 &
python  ${path}/main.py 11129 &
python  ${path}/main.py 11130 &
python  ${path}/main.py 11131 &
python  ${path}/main.py 11132 &
python  ${path}/main.py 11133 &
python  ${path}/main.py 11134 &
python  ${path}/main.py 11135 &
python  ${path}/main.py 11136 &
python  ${path}/main.py 11137 &
python  ${path}/main.py 11138 &
python  ${path}/main.py 11139 &
python  ${path}/main.py 11140 &
python  ${path}/main.py 11141 &
python  ${path}/main.py 11142 &
python  ${path}/main.py 11143 &
python  ${path}/main.py 11144 &
python  ${path}/main.py 11145 &
python  ${path}/main.py 11146 &
python  ${path}/main.py 11147 &
python  ${path}/main.py 11148 &
python  ${path}/main.py 11149 &
python  ${path}/main.py 11150 &
python  ${path}/main.py 11151 &
python  ${path}/main.py 11152 &
python  ${path}/main.py 11153 &
python  ${path}/main.py 11154 &
python  ${path}/main.py 11155 &
python  ${path}/main.py 11156 &
python  ${path}/main.py 11157 &
python  ${path}/main.py 11158 &
python  ${path}/main.py 11159 &
wait
