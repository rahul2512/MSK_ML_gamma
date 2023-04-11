#!/bin/bash -l
#SBATCH -J RNN_9
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 288 &
python  ${path}/main.py 289 &
python  ${path}/main.py 290 &
python  ${path}/main.py 291 &
python  ${path}/main.py 292 &
python  ${path}/main.py 293 &
python  ${path}/main.py 294 &
python  ${path}/main.py 295 &
python  ${path}/main.py 296 &
python  ${path}/main.py 297 &
python  ${path}/main.py 298 &
python  ${path}/main.py 299 &
python  ${path}/main.py 300 &
python  ${path}/main.py 301 &
python  ${path}/main.py 302 &
python  ${path}/main.py 303 &
python  ${path}/main.py 304 &
python  ${path}/main.py 305 &
python  ${path}/main.py 306 &
python  ${path}/main.py 307 &
python  ${path}/main.py 308 &
python  ${path}/main.py 309 &
python  ${path}/main.py 310 &
python  ${path}/main.py 311 &
python  ${path}/main.py 312 &
python  ${path}/main.py 313 &
python  ${path}/main.py 314 &
python  ${path}/main.py 315 &
python  ${path}/main.py 316 &
python  ${path}/main.py 317 &
python  ${path}/main.py 318 &
python  ${path}/main.py 319 &
python  ${path}/main.py 320 &
python  ${path}/main.py 321 &
python  ${path}/main.py 322 &
python  ${path}/main.py 323 &
wait
