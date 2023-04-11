#!/bin/bash -l
#SBATCH -J RNN_8
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 252 &
python  ${path}/main.py 253 &
python  ${path}/main.py 254 &
python  ${path}/main.py 255 &
python  ${path}/main.py 256 &
python  ${path}/main.py 257 &
python  ${path}/main.py 258 &
python  ${path}/main.py 259 &
python  ${path}/main.py 260 &
python  ${path}/main.py 261 &
python  ${path}/main.py 262 &
python  ${path}/main.py 263 &
python  ${path}/main.py 264 &
python  ${path}/main.py 265 &
python  ${path}/main.py 266 &
python  ${path}/main.py 267 &
python  ${path}/main.py 268 &
python  ${path}/main.py 269 &
python  ${path}/main.py 270 &
python  ${path}/main.py 271 &
python  ${path}/main.py 272 &
python  ${path}/main.py 273 &
python  ${path}/main.py 274 &
python  ${path}/main.py 275 &
python  ${path}/main.py 276 &
python  ${path}/main.py 277 &
python  ${path}/main.py 278 &
python  ${path}/main.py 279 &
python  ${path}/main.py 280 &
python  ${path}/main.py 281 &
python  ${path}/main.py 282 &
python  ${path}/main.py 283 &
python  ${path}/main.py 284 &
python  ${path}/main.py 285 &
python  ${path}/main.py 286 &
python  ${path}/main.py 287 &
wait
