#!/bin/bash -l
#SBATCH -J NN_117
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8352 &
python  ${path}/main.py 8353 &
python  ${path}/main.py 8354 &
python  ${path}/main.py 8355 &
python  ${path}/main.py 8356 &
python  ${path}/main.py 8357 &
python  ${path}/main.py 8358 &
python  ${path}/main.py 8359 &
python  ${path}/main.py 8360 &
python  ${path}/main.py 8361 &
python  ${path}/main.py 8362 &
python  ${path}/main.py 8363 &
python  ${path}/main.py 8364 &
python  ${path}/main.py 8365 &
python  ${path}/main.py 8366 &
python  ${path}/main.py 8367 &
python  ${path}/main.py 8368 &
python  ${path}/main.py 8369 &
python  ${path}/main.py 8370 &
python  ${path}/main.py 8371 &
python  ${path}/main.py 8372 &
python  ${path}/main.py 8373 &
python  ${path}/main.py 8374 &
python  ${path}/main.py 8375 &
python  ${path}/main.py 8376 &
python  ${path}/main.py 8377 &
python  ${path}/main.py 8378 &
python  ${path}/main.py 8379 &
python  ${path}/main.py 8380 &
python  ${path}/main.py 8381 &
python  ${path}/main.py 8382 &
python  ${path}/main.py 8383 &
python  ${path}/main.py 8384 &
python  ${path}/main.py 8385 &
python  ${path}/main.py 8386 &
python  ${path}/main.py 8387 &
python  ${path}/main.py 8388 &
python  ${path}/main.py 8389 &
python  ${path}/main.py 8390 &
python  ${path}/main.py 8391 &
python  ${path}/main.py 8392 &
python  ${path}/main.py 8393 &
python  ${path}/main.py 8394 &
python  ${path}/main.py 8395 &
python  ${path}/main.py 8396 &
python  ${path}/main.py 8397 &
python  ${path}/main.py 8398 &
python  ${path}/main.py 8399 &
python  ${path}/main.py 8400 &
python  ${path}/main.py 8401 &
python  ${path}/main.py 8402 &
python  ${path}/main.py 8403 &
python  ${path}/main.py 8404 &
python  ${path}/main.py 8405 &
python  ${path}/main.py 8406 &
python  ${path}/main.py 8407 &
python  ${path}/main.py 8408 &
python  ${path}/main.py 8409 &
python  ${path}/main.py 8410 &
python  ${path}/main.py 8411 &
python  ${path}/main.py 8412 &
python  ${path}/main.py 8413 &
python  ${path}/main.py 8414 &
python  ${path}/main.py 8415 &
python  ${path}/main.py 8416 &
python  ${path}/main.py 8417 &
python  ${path}/main.py 8418 &
python  ${path}/main.py 8419 &
python  ${path}/main.py 8420 &
python  ${path}/main.py 8421 &
python  ${path}/main.py 8422 &
python  ${path}/main.py 8423 &
wait
