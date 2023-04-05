#!/bin/bash -l
#SBATCH -J NN_105
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7488 &
python  ${path}/main.py 7489 &
python  ${path}/main.py 7490 &
python  ${path}/main.py 7491 &
python  ${path}/main.py 7492 &
python  ${path}/main.py 7493 &
python  ${path}/main.py 7494 &
python  ${path}/main.py 7495 &
python  ${path}/main.py 7496 &
python  ${path}/main.py 7497 &
python  ${path}/main.py 7498 &
python  ${path}/main.py 7499 &
python  ${path}/main.py 7500 &
python  ${path}/main.py 7501 &
python  ${path}/main.py 7502 &
python  ${path}/main.py 7503 &
python  ${path}/main.py 7504 &
python  ${path}/main.py 7505 &
python  ${path}/main.py 7506 &
python  ${path}/main.py 7507 &
python  ${path}/main.py 7508 &
python  ${path}/main.py 7509 &
python  ${path}/main.py 7510 &
python  ${path}/main.py 7511 &
python  ${path}/main.py 7512 &
python  ${path}/main.py 7513 &
python  ${path}/main.py 7514 &
python  ${path}/main.py 7515 &
python  ${path}/main.py 7516 &
python  ${path}/main.py 7517 &
python  ${path}/main.py 7518 &
python  ${path}/main.py 7519 &
python  ${path}/main.py 7520 &
python  ${path}/main.py 7521 &
python  ${path}/main.py 7522 &
python  ${path}/main.py 7523 &
python  ${path}/main.py 7524 &
python  ${path}/main.py 7525 &
python  ${path}/main.py 7526 &
python  ${path}/main.py 7527 &
python  ${path}/main.py 7528 &
python  ${path}/main.py 7529 &
python  ${path}/main.py 7530 &
python  ${path}/main.py 7531 &
python  ${path}/main.py 7532 &
python  ${path}/main.py 7533 &
python  ${path}/main.py 7534 &
python  ${path}/main.py 7535 &
python  ${path}/main.py 7536 &
python  ${path}/main.py 7537 &
python  ${path}/main.py 7538 &
python  ${path}/main.py 7539 &
python  ${path}/main.py 7540 &
python  ${path}/main.py 7541 &
python  ${path}/main.py 7542 &
python  ${path}/main.py 7543 &
python  ${path}/main.py 7544 &
python  ${path}/main.py 7545 &
python  ${path}/main.py 7546 &
python  ${path}/main.py 7547 &
python  ${path}/main.py 7548 &
python  ${path}/main.py 7549 &
python  ${path}/main.py 7550 &
python  ${path}/main.py 7551 &
python  ${path}/main.py 7552 &
python  ${path}/main.py 7553 &
python  ${path}/main.py 7554 &
python  ${path}/main.py 7555 &
python  ${path}/main.py 7556 &
python  ${path}/main.py 7557 &
python  ${path}/main.py 7558 &
python  ${path}/main.py 7559 &
wait
