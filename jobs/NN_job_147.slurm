#!/bin/bash -l
#SBATCH -J NN_147
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10512 &
python  ${path}/main.py 10513 &
python  ${path}/main.py 10514 &
python  ${path}/main.py 10515 &
python  ${path}/main.py 10516 &
python  ${path}/main.py 10517 &
python  ${path}/main.py 10518 &
python  ${path}/main.py 10519 &
python  ${path}/main.py 10520 &
python  ${path}/main.py 10521 &
python  ${path}/main.py 10522 &
python  ${path}/main.py 10523 &
python  ${path}/main.py 10524 &
python  ${path}/main.py 10525 &
python  ${path}/main.py 10526 &
python  ${path}/main.py 10527 &
python  ${path}/main.py 10528 &
python  ${path}/main.py 10529 &
python  ${path}/main.py 10530 &
python  ${path}/main.py 10531 &
python  ${path}/main.py 10532 &
python  ${path}/main.py 10533 &
python  ${path}/main.py 10534 &
python  ${path}/main.py 10535 &
python  ${path}/main.py 10536 &
python  ${path}/main.py 10537 &
python  ${path}/main.py 10538 &
python  ${path}/main.py 10539 &
python  ${path}/main.py 10540 &
python  ${path}/main.py 10541 &
python  ${path}/main.py 10542 &
python  ${path}/main.py 10543 &
python  ${path}/main.py 10544 &
python  ${path}/main.py 10545 &
python  ${path}/main.py 10546 &
python  ${path}/main.py 10547 &
python  ${path}/main.py 10548 &
python  ${path}/main.py 10549 &
python  ${path}/main.py 10550 &
python  ${path}/main.py 10551 &
python  ${path}/main.py 10552 &
python  ${path}/main.py 10553 &
python  ${path}/main.py 10554 &
python  ${path}/main.py 10555 &
python  ${path}/main.py 10556 &
python  ${path}/main.py 10557 &
python  ${path}/main.py 10558 &
python  ${path}/main.py 10559 &
python  ${path}/main.py 10560 &
python  ${path}/main.py 10561 &
python  ${path}/main.py 10562 &
python  ${path}/main.py 10563 &
python  ${path}/main.py 10564 &
python  ${path}/main.py 10565 &
python  ${path}/main.py 10566 &
python  ${path}/main.py 10567 &
python  ${path}/main.py 10568 &
python  ${path}/main.py 10569 &
python  ${path}/main.py 10570 &
python  ${path}/main.py 10571 &
python  ${path}/main.py 10572 &
python  ${path}/main.py 10573 &
python  ${path}/main.py 10574 &
python  ${path}/main.py 10575 &
python  ${path}/main.py 10576 &
python  ${path}/main.py 10577 &
python  ${path}/main.py 10578 &
python  ${path}/main.py 10579 &
python  ${path}/main.py 10580 &
python  ${path}/main.py 10581 &
python  ${path}/main.py 10582 &
python  ${path}/main.py 10583 &
wait
