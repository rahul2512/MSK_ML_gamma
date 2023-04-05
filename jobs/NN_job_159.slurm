#!/bin/bash -l
#SBATCH -J NN_159
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11376 &
python  ${path}/main.py 11377 &
python  ${path}/main.py 11378 &
python  ${path}/main.py 11379 &
python  ${path}/main.py 11380 &
python  ${path}/main.py 11381 &
python  ${path}/main.py 11382 &
python  ${path}/main.py 11383 &
python  ${path}/main.py 11384 &
python  ${path}/main.py 11385 &
python  ${path}/main.py 11386 &
python  ${path}/main.py 11387 &
python  ${path}/main.py 11388 &
python  ${path}/main.py 11389 &
python  ${path}/main.py 11390 &
python  ${path}/main.py 11391 &
python  ${path}/main.py 11392 &
python  ${path}/main.py 11393 &
python  ${path}/main.py 11394 &
python  ${path}/main.py 11395 &
python  ${path}/main.py 11396 &
python  ${path}/main.py 11397 &
python  ${path}/main.py 11398 &
python  ${path}/main.py 11399 &
python  ${path}/main.py 11400 &
python  ${path}/main.py 11401 &
python  ${path}/main.py 11402 &
python  ${path}/main.py 11403 &
python  ${path}/main.py 11404 &
python  ${path}/main.py 11405 &
python  ${path}/main.py 11406 &
python  ${path}/main.py 11407 &
python  ${path}/main.py 11408 &
python  ${path}/main.py 11409 &
python  ${path}/main.py 11410 &
python  ${path}/main.py 11411 &
python  ${path}/main.py 11412 &
python  ${path}/main.py 11413 &
python  ${path}/main.py 11414 &
python  ${path}/main.py 11415 &
python  ${path}/main.py 11416 &
python  ${path}/main.py 11417 &
python  ${path}/main.py 11418 &
python  ${path}/main.py 11419 &
python  ${path}/main.py 11420 &
python  ${path}/main.py 11421 &
python  ${path}/main.py 11422 &
python  ${path}/main.py 11423 &
python  ${path}/main.py 11424 &
python  ${path}/main.py 11425 &
python  ${path}/main.py 11426 &
python  ${path}/main.py 11427 &
python  ${path}/main.py 11428 &
python  ${path}/main.py 11429 &
python  ${path}/main.py 11430 &
python  ${path}/main.py 11431 &
python  ${path}/main.py 11432 &
python  ${path}/main.py 11433 &
python  ${path}/main.py 11434 &
python  ${path}/main.py 11435 &
python  ${path}/main.py 11436 &
python  ${path}/main.py 11437 &
python  ${path}/main.py 11438 &
python  ${path}/main.py 11439 &
python  ${path}/main.py 11440 &
python  ${path}/main.py 11441 &
python  ${path}/main.py 11442 &
python  ${path}/main.py 11443 &
python  ${path}/main.py 11444 &
python  ${path}/main.py 11445 &
python  ${path}/main.py 11446 &
python  ${path}/main.py 11447 &
wait
