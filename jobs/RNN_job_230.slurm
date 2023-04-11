#!/bin/bash -l
#SBATCH -J RNN_230
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8244 &
python  ${path}/main.py 8245 &
python  ${path}/main.py 8246 &
python  ${path}/main.py 8247 &
python  ${path}/main.py 8248 &
python  ${path}/main.py 8249 &
python  ${path}/main.py 8250 &
python  ${path}/main.py 8251 &
python  ${path}/main.py 8252 &
python  ${path}/main.py 8253 &
python  ${path}/main.py 8254 &
python  ${path}/main.py 8255 &
python  ${path}/main.py 8256 &
python  ${path}/main.py 8257 &
python  ${path}/main.py 8258 &
python  ${path}/main.py 8259 &
python  ${path}/main.py 8260 &
python  ${path}/main.py 8261 &
python  ${path}/main.py 8262 &
python  ${path}/main.py 8263 &
python  ${path}/main.py 8264 &
python  ${path}/main.py 8265 &
python  ${path}/main.py 8266 &
python  ${path}/main.py 8267 &
python  ${path}/main.py 8268 &
python  ${path}/main.py 8269 &
python  ${path}/main.py 8270 &
python  ${path}/main.py 8271 &
python  ${path}/main.py 8272 &
python  ${path}/main.py 8273 &
python  ${path}/main.py 8274 &
python  ${path}/main.py 8275 &
python  ${path}/main.py 8276 &
python  ${path}/main.py 8277 &
python  ${path}/main.py 8278 &
python  ${path}/main.py 8279 &
wait
