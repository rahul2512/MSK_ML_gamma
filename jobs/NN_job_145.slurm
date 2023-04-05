#!/bin/bash -l
#SBATCH -J NN_145
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10368 &
python  ${path}/main.py 10369 &
python  ${path}/main.py 10370 &
python  ${path}/main.py 10371 &
python  ${path}/main.py 10372 &
python  ${path}/main.py 10373 &
python  ${path}/main.py 10374 &
python  ${path}/main.py 10375 &
python  ${path}/main.py 10376 &
python  ${path}/main.py 10377 &
python  ${path}/main.py 10378 &
python  ${path}/main.py 10379 &
python  ${path}/main.py 10380 &
python  ${path}/main.py 10381 &
python  ${path}/main.py 10382 &
python  ${path}/main.py 10383 &
python  ${path}/main.py 10384 &
python  ${path}/main.py 10385 &
python  ${path}/main.py 10386 &
python  ${path}/main.py 10387 &
python  ${path}/main.py 10388 &
python  ${path}/main.py 10389 &
python  ${path}/main.py 10390 &
python  ${path}/main.py 10391 &
python  ${path}/main.py 10392 &
python  ${path}/main.py 10393 &
python  ${path}/main.py 10394 &
python  ${path}/main.py 10395 &
python  ${path}/main.py 10396 &
python  ${path}/main.py 10397 &
python  ${path}/main.py 10398 &
python  ${path}/main.py 10399 &
python  ${path}/main.py 10400 &
python  ${path}/main.py 10401 &
python  ${path}/main.py 10402 &
python  ${path}/main.py 10403 &
python  ${path}/main.py 10404 &
python  ${path}/main.py 10405 &
python  ${path}/main.py 10406 &
python  ${path}/main.py 10407 &
python  ${path}/main.py 10408 &
python  ${path}/main.py 10409 &
python  ${path}/main.py 10410 &
python  ${path}/main.py 10411 &
python  ${path}/main.py 10412 &
python  ${path}/main.py 10413 &
python  ${path}/main.py 10414 &
python  ${path}/main.py 10415 &
python  ${path}/main.py 10416 &
python  ${path}/main.py 10417 &
python  ${path}/main.py 10418 &
python  ${path}/main.py 10419 &
python  ${path}/main.py 10420 &
python  ${path}/main.py 10421 &
python  ${path}/main.py 10422 &
python  ${path}/main.py 10423 &
python  ${path}/main.py 10424 &
python  ${path}/main.py 10425 &
python  ${path}/main.py 10426 &
python  ${path}/main.py 10427 &
python  ${path}/main.py 10428 &
python  ${path}/main.py 10429 &
python  ${path}/main.py 10430 &
python  ${path}/main.py 10431 &
python  ${path}/main.py 10432 &
python  ${path}/main.py 10433 &
python  ${path}/main.py 10434 &
python  ${path}/main.py 10435 &
python  ${path}/main.py 10436 &
python  ${path}/main.py 10437 &
python  ${path}/main.py 10438 &
python  ${path}/main.py 10439 &
wait
