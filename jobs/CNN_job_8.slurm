#!/bin/bash -l
#SBATCH -J CNN_8
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 168 &
python  ${path}/main.py 169 &
python  ${path}/main.py 170 &
python  ${path}/main.py 171 &
python  ${path}/main.py 172 &
python  ${path}/main.py 173 &
python  ${path}/main.py 174 &
python  ${path}/main.py 175 &
python  ${path}/main.py 176 &
python  ${path}/main.py 177 &
python  ${path}/main.py 178 &
python  ${path}/main.py 179 &
python  ${path}/main.py 180 &
python  ${path}/main.py 181 &
python  ${path}/main.py 182 &
python  ${path}/main.py 183 &
python  ${path}/main.py 184 &
python  ${path}/main.py 185 &
python  ${path}/main.py 186 &
python  ${path}/main.py 187 &
python  ${path}/main.py 188 &
python  ${path}/main.py 189 &
python  ${path}/main.py 190 &
python  ${path}/main.py 191 &
wait
