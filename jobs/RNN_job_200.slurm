#!/bin/bash -l
#SBATCH -J RNN_200
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7164 &
python  ${path}/main.py 7165 &
python  ${path}/main.py 7166 &
python  ${path}/main.py 7167 &
python  ${path}/main.py 7168 &
python  ${path}/main.py 7169 &
python  ${path}/main.py 7170 &
python  ${path}/main.py 7171 &
python  ${path}/main.py 7172 &
python  ${path}/main.py 7173 &
python  ${path}/main.py 7174 &
python  ${path}/main.py 7175 &
python  ${path}/main.py 7176 &
python  ${path}/main.py 7177 &
python  ${path}/main.py 7178 &
python  ${path}/main.py 7179 &
python  ${path}/main.py 7180 &
python  ${path}/main.py 7181 &
python  ${path}/main.py 7182 &
python  ${path}/main.py 7183 &
python  ${path}/main.py 7184 &
python  ${path}/main.py 7185 &
python  ${path}/main.py 7186 &
python  ${path}/main.py 7187 &
python  ${path}/main.py 7188 &
python  ${path}/main.py 7189 &
python  ${path}/main.py 7190 &
python  ${path}/main.py 7191 &
python  ${path}/main.py 7192 &
python  ${path}/main.py 7193 &
python  ${path}/main.py 7194 &
python  ${path}/main.py 7195 &
python  ${path}/main.py 7196 &
python  ${path}/main.py 7197 &
python  ${path}/main.py 7198 &
python  ${path}/main.py 7199 &
wait
