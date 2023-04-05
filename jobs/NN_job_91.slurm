#!/bin/bash -l
#SBATCH -J NN_91
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 6480 &
python  ${path}/main.py 6481 &
python  ${path}/main.py 6482 &
python  ${path}/main.py 6483 &
python  ${path}/main.py 6484 &
python  ${path}/main.py 6485 &
python  ${path}/main.py 6486 &
python  ${path}/main.py 6487 &
python  ${path}/main.py 6488 &
python  ${path}/main.py 6489 &
python  ${path}/main.py 6490 &
python  ${path}/main.py 6491 &
python  ${path}/main.py 6492 &
python  ${path}/main.py 6493 &
python  ${path}/main.py 6494 &
python  ${path}/main.py 6495 &
python  ${path}/main.py 6496 &
python  ${path}/main.py 6497 &
python  ${path}/main.py 6498 &
python  ${path}/main.py 6499 &
python  ${path}/main.py 6500 &
python  ${path}/main.py 6501 &
python  ${path}/main.py 6502 &
python  ${path}/main.py 6503 &
python  ${path}/main.py 6504 &
python  ${path}/main.py 6505 &
python  ${path}/main.py 6506 &
python  ${path}/main.py 6507 &
python  ${path}/main.py 6508 &
python  ${path}/main.py 6509 &
python  ${path}/main.py 6510 &
python  ${path}/main.py 6511 &
python  ${path}/main.py 6512 &
python  ${path}/main.py 6513 &
python  ${path}/main.py 6514 &
python  ${path}/main.py 6515 &
python  ${path}/main.py 6516 &
python  ${path}/main.py 6517 &
python  ${path}/main.py 6518 &
python  ${path}/main.py 6519 &
python  ${path}/main.py 6520 &
python  ${path}/main.py 6521 &
python  ${path}/main.py 6522 &
python  ${path}/main.py 6523 &
python  ${path}/main.py 6524 &
python  ${path}/main.py 6525 &
python  ${path}/main.py 6526 &
python  ${path}/main.py 6527 &
python  ${path}/main.py 6528 &
python  ${path}/main.py 6529 &
python  ${path}/main.py 6530 &
python  ${path}/main.py 6531 &
python  ${path}/main.py 6532 &
python  ${path}/main.py 6533 &
python  ${path}/main.py 6534 &
python  ${path}/main.py 6535 &
python  ${path}/main.py 6536 &
python  ${path}/main.py 6537 &
python  ${path}/main.py 6538 &
python  ${path}/main.py 6539 &
python  ${path}/main.py 6540 &
python  ${path}/main.py 6541 &
python  ${path}/main.py 6542 &
python  ${path}/main.py 6543 &
python  ${path}/main.py 6544 &
python  ${path}/main.py 6545 &
python  ${path}/main.py 6546 &
python  ${path}/main.py 6547 &
python  ${path}/main.py 6548 &
python  ${path}/main.py 6549 &
python  ${path}/main.py 6550 &
python  ${path}/main.py 6551 &
wait
