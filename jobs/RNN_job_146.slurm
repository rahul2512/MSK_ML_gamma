#!/bin/bash -l
#SBATCH -J RNN_146
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5220 &
python  ${path}/main.py 5221 &
python  ${path}/main.py 5222 &
python  ${path}/main.py 5223 &
python  ${path}/main.py 5224 &
python  ${path}/main.py 5225 &
python  ${path}/main.py 5226 &
python  ${path}/main.py 5227 &
python  ${path}/main.py 5228 &
python  ${path}/main.py 5229 &
python  ${path}/main.py 5230 &
python  ${path}/main.py 5231 &
python  ${path}/main.py 5232 &
python  ${path}/main.py 5233 &
python  ${path}/main.py 5234 &
python  ${path}/main.py 5235 &
python  ${path}/main.py 5236 &
python  ${path}/main.py 5237 &
python  ${path}/main.py 5238 &
python  ${path}/main.py 5239 &
python  ${path}/main.py 5240 &
python  ${path}/main.py 5241 &
python  ${path}/main.py 5242 &
python  ${path}/main.py 5243 &
python  ${path}/main.py 5244 &
python  ${path}/main.py 5245 &
python  ${path}/main.py 5246 &
python  ${path}/main.py 5247 &
python  ${path}/main.py 5248 &
python  ${path}/main.py 5249 &
python  ${path}/main.py 5250 &
python  ${path}/main.py 5251 &
python  ${path}/main.py 5252 &
python  ${path}/main.py 5253 &
python  ${path}/main.py 5254 &
python  ${path}/main.py 5255 &
wait
