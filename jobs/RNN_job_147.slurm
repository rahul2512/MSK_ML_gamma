#!/bin/bash -l
#SBATCH -J RNN_147
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5256 &
python  ${path}/main.py 5257 &
python  ${path}/main.py 5258 &
python  ${path}/main.py 5259 &
python  ${path}/main.py 5260 &
python  ${path}/main.py 5261 &
python  ${path}/main.py 5262 &
python  ${path}/main.py 5263 &
python  ${path}/main.py 5264 &
python  ${path}/main.py 5265 &
python  ${path}/main.py 5266 &
python  ${path}/main.py 5267 &
python  ${path}/main.py 5268 &
python  ${path}/main.py 5269 &
python  ${path}/main.py 5270 &
python  ${path}/main.py 5271 &
python  ${path}/main.py 5272 &
python  ${path}/main.py 5273 &
python  ${path}/main.py 5274 &
python  ${path}/main.py 5275 &
python  ${path}/main.py 5276 &
python  ${path}/main.py 5277 &
python  ${path}/main.py 5278 &
python  ${path}/main.py 5279 &
python  ${path}/main.py 5280 &
python  ${path}/main.py 5281 &
python  ${path}/main.py 5282 &
python  ${path}/main.py 5283 &
python  ${path}/main.py 5284 &
python  ${path}/main.py 5285 &
python  ${path}/main.py 5286 &
python  ${path}/main.py 5287 &
python  ${path}/main.py 5288 &
python  ${path}/main.py 5289 &
python  ${path}/main.py 5290 &
python  ${path}/main.py 5291 &
wait
