#!/bin/bash -l
#SBATCH -J NN_158
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11304 &
python  ${path}/main.py 11305 &
python  ${path}/main.py 11306 &
python  ${path}/main.py 11307 &
python  ${path}/main.py 11308 &
python  ${path}/main.py 11309 &
python  ${path}/main.py 11310 &
python  ${path}/main.py 11311 &
python  ${path}/main.py 11312 &
python  ${path}/main.py 11313 &
python  ${path}/main.py 11314 &
python  ${path}/main.py 11315 &
python  ${path}/main.py 11316 &
python  ${path}/main.py 11317 &
python  ${path}/main.py 11318 &
python  ${path}/main.py 11319 &
python  ${path}/main.py 11320 &
python  ${path}/main.py 11321 &
python  ${path}/main.py 11322 &
python  ${path}/main.py 11323 &
python  ${path}/main.py 11324 &
python  ${path}/main.py 11325 &
python  ${path}/main.py 11326 &
python  ${path}/main.py 11327 &
python  ${path}/main.py 11328 &
python  ${path}/main.py 11329 &
python  ${path}/main.py 11330 &
python  ${path}/main.py 11331 &
python  ${path}/main.py 11332 &
python  ${path}/main.py 11333 &
python  ${path}/main.py 11334 &
python  ${path}/main.py 11335 &
python  ${path}/main.py 11336 &
python  ${path}/main.py 11337 &
python  ${path}/main.py 11338 &
python  ${path}/main.py 11339 &
python  ${path}/main.py 11340 &
python  ${path}/main.py 11341 &
python  ${path}/main.py 11342 &
python  ${path}/main.py 11343 &
python  ${path}/main.py 11344 &
python  ${path}/main.py 11345 &
python  ${path}/main.py 11346 &
python  ${path}/main.py 11347 &
python  ${path}/main.py 11348 &
python  ${path}/main.py 11349 &
python  ${path}/main.py 11350 &
python  ${path}/main.py 11351 &
python  ${path}/main.py 11352 &
python  ${path}/main.py 11353 &
python  ${path}/main.py 11354 &
python  ${path}/main.py 11355 &
python  ${path}/main.py 11356 &
python  ${path}/main.py 11357 &
python  ${path}/main.py 11358 &
python  ${path}/main.py 11359 &
python  ${path}/main.py 11360 &
python  ${path}/main.py 11361 &
python  ${path}/main.py 11362 &
python  ${path}/main.py 11363 &
python  ${path}/main.py 11364 &
python  ${path}/main.py 11365 &
python  ${path}/main.py 11366 &
python  ${path}/main.py 11367 &
python  ${path}/main.py 11368 &
python  ${path}/main.py 11369 &
python  ${path}/main.py 11370 &
python  ${path}/main.py 11371 &
python  ${path}/main.py 11372 &
python  ${path}/main.py 11373 &
python  ${path}/main.py 11374 &
python  ${path}/main.py 11375 &
wait
