#!/bin/bash -l
#SBATCH -J RNN_228
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
