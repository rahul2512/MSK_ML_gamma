#!/bin/bash -l
#SBATCH -J NN_114
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8136 &
python  ${path}/main.py 8137 &
python  ${path}/main.py 8138 &
python  ${path}/main.py 8139 &
python  ${path}/main.py 8140 &
python  ${path}/main.py 8141 &
python  ${path}/main.py 8142 &
python  ${path}/main.py 8143 &
python  ${path}/main.py 8144 &
python  ${path}/main.py 8145 &
python  ${path}/main.py 8146 &
python  ${path}/main.py 8147 &
python  ${path}/main.py 8148 &
python  ${path}/main.py 8149 &
python  ${path}/main.py 8150 &
python  ${path}/main.py 8151 &
python  ${path}/main.py 8152 &
python  ${path}/main.py 8153 &
python  ${path}/main.py 8154 &
python  ${path}/main.py 8155 &
python  ${path}/main.py 8156 &
python  ${path}/main.py 8157 &
python  ${path}/main.py 8158 &
python  ${path}/main.py 8159 &
python  ${path}/main.py 8160 &
python  ${path}/main.py 8161 &
python  ${path}/main.py 8162 &
python  ${path}/main.py 8163 &
python  ${path}/main.py 8164 &
python  ${path}/main.py 8165 &
python  ${path}/main.py 8166 &
python  ${path}/main.py 8167 &
python  ${path}/main.py 8168 &
python  ${path}/main.py 8169 &
python  ${path}/main.py 8170 &
python  ${path}/main.py 8171 &
python  ${path}/main.py 8172 &
python  ${path}/main.py 8173 &
python  ${path}/main.py 8174 &
python  ${path}/main.py 8175 &
python  ${path}/main.py 8176 &
python  ${path}/main.py 8177 &
python  ${path}/main.py 8178 &
python  ${path}/main.py 8179 &
python  ${path}/main.py 8180 &
python  ${path}/main.py 8181 &
python  ${path}/main.py 8182 &
python  ${path}/main.py 8183 &
python  ${path}/main.py 8184 &
python  ${path}/main.py 8185 &
python  ${path}/main.py 8186 &
python  ${path}/main.py 8187 &
python  ${path}/main.py 8188 &
python  ${path}/main.py 8189 &
python  ${path}/main.py 8190 &
python  ${path}/main.py 8191 &
python  ${path}/main.py 8192 &
python  ${path}/main.py 8193 &
python  ${path}/main.py 8194 &
python  ${path}/main.py 8195 &
python  ${path}/main.py 8196 &
python  ${path}/main.py 8197 &
python  ${path}/main.py 8198 &
python  ${path}/main.py 8199 &
python  ${path}/main.py 8200 &
python  ${path}/main.py 8201 &
python  ${path}/main.py 8202 &
python  ${path}/main.py 8203 &
python  ${path}/main.py 8204 &
python  ${path}/main.py 8205 &
python  ${path}/main.py 8206 &
python  ${path}/main.py 8207 &
wait
