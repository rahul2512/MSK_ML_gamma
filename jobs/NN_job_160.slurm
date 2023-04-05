#!/bin/bash -l
#SBATCH -J NN_160
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11448 &
python  ${path}/main.py 11449 &
python  ${path}/main.py 11450 &
python  ${path}/main.py 11451 &
python  ${path}/main.py 11452 &
python  ${path}/main.py 11453 &
python  ${path}/main.py 11454 &
python  ${path}/main.py 11455 &
python  ${path}/main.py 11456 &
python  ${path}/main.py 11457 &
python  ${path}/main.py 11458 &
python  ${path}/main.py 11459 &
python  ${path}/main.py 11460 &
python  ${path}/main.py 11461 &
python  ${path}/main.py 11462 &
python  ${path}/main.py 11463 &
python  ${path}/main.py 11464 &
python  ${path}/main.py 11465 &
python  ${path}/main.py 11466 &
python  ${path}/main.py 11467 &
python  ${path}/main.py 11468 &
python  ${path}/main.py 11469 &
python  ${path}/main.py 11470 &
python  ${path}/main.py 11471 &
python  ${path}/main.py 11472 &
python  ${path}/main.py 11473 &
python  ${path}/main.py 11474 &
python  ${path}/main.py 11475 &
python  ${path}/main.py 11476 &
python  ${path}/main.py 11477 &
python  ${path}/main.py 11478 &
python  ${path}/main.py 11479 &
python  ${path}/main.py 11480 &
python  ${path}/main.py 11481 &
python  ${path}/main.py 11482 &
python  ${path}/main.py 11483 &
python  ${path}/main.py 11484 &
python  ${path}/main.py 11485 &
python  ${path}/main.py 11486 &
python  ${path}/main.py 11487 &
python  ${path}/main.py 11488 &
python  ${path}/main.py 11489 &
python  ${path}/main.py 11490 &
python  ${path}/main.py 11491 &
python  ${path}/main.py 11492 &
python  ${path}/main.py 11493 &
python  ${path}/main.py 11494 &
python  ${path}/main.py 11495 &
python  ${path}/main.py 11496 &
python  ${path}/main.py 11497 &
python  ${path}/main.py 11498 &
python  ${path}/main.py 11499 &
python  ${path}/main.py 11500 &
python  ${path}/main.py 11501 &
python  ${path}/main.py 11502 &
python  ${path}/main.py 11503 &
python  ${path}/main.py 11504 &
python  ${path}/main.py 11505 &
python  ${path}/main.py 11506 &
python  ${path}/main.py 11507 &
python  ${path}/main.py 11508 &
python  ${path}/main.py 11509 &
python  ${path}/main.py 11510 &
python  ${path}/main.py 11511 &
python  ${path}/main.py 11512 &
python  ${path}/main.py 11513 &
python  ${path}/main.py 11514 &
python  ${path}/main.py 11515 &
python  ${path}/main.py 11516 &
python  ${path}/main.py 11517 &
python  ${path}/main.py 11518 &
python  ${path}/main.py 11519 &
wait
