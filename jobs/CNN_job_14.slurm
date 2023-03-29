#!/bin/bash -l
#SBATCH -J CNN_14
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
python  ${path}/main.py 324 &
python  ${path}/main.py 325 &
python  ${path}/main.py 326 &
python  ${path}/main.py 327 &
python  ${path}/main.py 328 &
python  ${path}/main.py 329 &
python  ${path}/main.py 330 &
python  ${path}/main.py 331 &
python  ${path}/main.py 332 &
python  ${path}/main.py 333 &
python  ${path}/main.py 334 &
python  ${path}/main.py 335 &
wait
