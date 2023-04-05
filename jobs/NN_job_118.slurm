#!/bin/bash -l
#SBATCH -J NN_118
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8424 &
python  ${path}/main.py 8425 &
python  ${path}/main.py 8426 &
python  ${path}/main.py 8427 &
python  ${path}/main.py 8428 &
python  ${path}/main.py 8429 &
python  ${path}/main.py 8430 &
python  ${path}/main.py 8431 &
python  ${path}/main.py 8432 &
python  ${path}/main.py 8433 &
python  ${path}/main.py 8434 &
python  ${path}/main.py 8435 &
python  ${path}/main.py 8436 &
python  ${path}/main.py 8437 &
python  ${path}/main.py 8438 &
python  ${path}/main.py 8439 &
python  ${path}/main.py 8440 &
python  ${path}/main.py 8441 &
python  ${path}/main.py 8442 &
python  ${path}/main.py 8443 &
python  ${path}/main.py 8444 &
python  ${path}/main.py 8445 &
python  ${path}/main.py 8446 &
python  ${path}/main.py 8447 &
python  ${path}/main.py 8448 &
python  ${path}/main.py 8449 &
python  ${path}/main.py 8450 &
python  ${path}/main.py 8451 &
python  ${path}/main.py 8452 &
python  ${path}/main.py 8453 &
python  ${path}/main.py 8454 &
python  ${path}/main.py 8455 &
python  ${path}/main.py 8456 &
python  ${path}/main.py 8457 &
python  ${path}/main.py 8458 &
python  ${path}/main.py 8459 &
python  ${path}/main.py 8460 &
python  ${path}/main.py 8461 &
python  ${path}/main.py 8462 &
python  ${path}/main.py 8463 &
python  ${path}/main.py 8464 &
python  ${path}/main.py 8465 &
python  ${path}/main.py 8466 &
python  ${path}/main.py 8467 &
python  ${path}/main.py 8468 &
python  ${path}/main.py 8469 &
python  ${path}/main.py 8470 &
python  ${path}/main.py 8471 &
python  ${path}/main.py 8472 &
python  ${path}/main.py 8473 &
python  ${path}/main.py 8474 &
python  ${path}/main.py 8475 &
python  ${path}/main.py 8476 &
python  ${path}/main.py 8477 &
python  ${path}/main.py 8478 &
python  ${path}/main.py 8479 &
python  ${path}/main.py 8480 &
python  ${path}/main.py 8481 &
python  ${path}/main.py 8482 &
python  ${path}/main.py 8483 &
python  ${path}/main.py 8484 &
python  ${path}/main.py 8485 &
python  ${path}/main.py 8486 &
python  ${path}/main.py 8487 &
python  ${path}/main.py 8488 &
python  ${path}/main.py 8489 &
python  ${path}/main.py 8490 &
python  ${path}/main.py 8491 &
python  ${path}/main.py 8492 &
python  ${path}/main.py 8493 &
python  ${path}/main.py 8494 &
python  ${path}/main.py 8495 &
wait
