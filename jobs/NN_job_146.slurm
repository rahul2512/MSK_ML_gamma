#!/bin/bash -l
#SBATCH -J NN_146
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10440 &
python  ${path}/main.py 10441 &
python  ${path}/main.py 10442 &
python  ${path}/main.py 10443 &
python  ${path}/main.py 10444 &
python  ${path}/main.py 10445 &
python  ${path}/main.py 10446 &
python  ${path}/main.py 10447 &
python  ${path}/main.py 10448 &
python  ${path}/main.py 10449 &
python  ${path}/main.py 10450 &
python  ${path}/main.py 10451 &
python  ${path}/main.py 10452 &
python  ${path}/main.py 10453 &
python  ${path}/main.py 10454 &
python  ${path}/main.py 10455 &
python  ${path}/main.py 10456 &
python  ${path}/main.py 10457 &
python  ${path}/main.py 10458 &
python  ${path}/main.py 10459 &
python  ${path}/main.py 10460 &
python  ${path}/main.py 10461 &
python  ${path}/main.py 10462 &
python  ${path}/main.py 10463 &
python  ${path}/main.py 10464 &
python  ${path}/main.py 10465 &
python  ${path}/main.py 10466 &
python  ${path}/main.py 10467 &
python  ${path}/main.py 10468 &
python  ${path}/main.py 10469 &
python  ${path}/main.py 10470 &
python  ${path}/main.py 10471 &
python  ${path}/main.py 10472 &
python  ${path}/main.py 10473 &
python  ${path}/main.py 10474 &
python  ${path}/main.py 10475 &
python  ${path}/main.py 10476 &
python  ${path}/main.py 10477 &
python  ${path}/main.py 10478 &
python  ${path}/main.py 10479 &
python  ${path}/main.py 10480 &
python  ${path}/main.py 10481 &
python  ${path}/main.py 10482 &
python  ${path}/main.py 10483 &
python  ${path}/main.py 10484 &
python  ${path}/main.py 10485 &
python  ${path}/main.py 10486 &
python  ${path}/main.py 10487 &
python  ${path}/main.py 10488 &
python  ${path}/main.py 10489 &
python  ${path}/main.py 10490 &
python  ${path}/main.py 10491 &
python  ${path}/main.py 10492 &
python  ${path}/main.py 10493 &
python  ${path}/main.py 10494 &
python  ${path}/main.py 10495 &
python  ${path}/main.py 10496 &
python  ${path}/main.py 10497 &
python  ${path}/main.py 10498 &
python  ${path}/main.py 10499 &
python  ${path}/main.py 10500 &
python  ${path}/main.py 10501 &
python  ${path}/main.py 10502 &
python  ${path}/main.py 10503 &
python  ${path}/main.py 10504 &
python  ${path}/main.py 10505 &
python  ${path}/main.py 10506 &
python  ${path}/main.py 10507 &
python  ${path}/main.py 10508 &
python  ${path}/main.py 10509 &
python  ${path}/main.py 10510 &
python  ${path}/main.py 10511 &
wait
