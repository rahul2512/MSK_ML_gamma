#!/bin/bash -l
#SBATCH -J NN_134
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9576 &
python  ${path}/main.py 9577 &
python  ${path}/main.py 9578 &
python  ${path}/main.py 9579 &
python  ${path}/main.py 9580 &
python  ${path}/main.py 9581 &
python  ${path}/main.py 9582 &
python  ${path}/main.py 9583 &
python  ${path}/main.py 9584 &
python  ${path}/main.py 9585 &
python  ${path}/main.py 9586 &
python  ${path}/main.py 9587 &
python  ${path}/main.py 9588 &
python  ${path}/main.py 9589 &
python  ${path}/main.py 9590 &
python  ${path}/main.py 9591 &
python  ${path}/main.py 9592 &
python  ${path}/main.py 9593 &
python  ${path}/main.py 9594 &
python  ${path}/main.py 9595 &
python  ${path}/main.py 9596 &
python  ${path}/main.py 9597 &
python  ${path}/main.py 9598 &
python  ${path}/main.py 9599 &
python  ${path}/main.py 9600 &
python  ${path}/main.py 9601 &
python  ${path}/main.py 9602 &
python  ${path}/main.py 9603 &
python  ${path}/main.py 9604 &
python  ${path}/main.py 9605 &
python  ${path}/main.py 9606 &
python  ${path}/main.py 9607 &
python  ${path}/main.py 9608 &
python  ${path}/main.py 9609 &
python  ${path}/main.py 9610 &
python  ${path}/main.py 9611 &
python  ${path}/main.py 9612 &
python  ${path}/main.py 9613 &
python  ${path}/main.py 9614 &
python  ${path}/main.py 9615 &
python  ${path}/main.py 9616 &
python  ${path}/main.py 9617 &
python  ${path}/main.py 9618 &
python  ${path}/main.py 9619 &
python  ${path}/main.py 9620 &
python  ${path}/main.py 9621 &
python  ${path}/main.py 9622 &
python  ${path}/main.py 9623 &
python  ${path}/main.py 9624 &
python  ${path}/main.py 9625 &
python  ${path}/main.py 9626 &
python  ${path}/main.py 9627 &
python  ${path}/main.py 9628 &
python  ${path}/main.py 9629 &
python  ${path}/main.py 9630 &
python  ${path}/main.py 9631 &
python  ${path}/main.py 9632 &
python  ${path}/main.py 9633 &
python  ${path}/main.py 9634 &
python  ${path}/main.py 9635 &
python  ${path}/main.py 9636 &
python  ${path}/main.py 9637 &
python  ${path}/main.py 9638 &
python  ${path}/main.py 9639 &
python  ${path}/main.py 9640 &
python  ${path}/main.py 9641 &
python  ${path}/main.py 9642 &
python  ${path}/main.py 9643 &
python  ${path}/main.py 9644 &
python  ${path}/main.py 9645 &
python  ${path}/main.py 9646 &
python  ${path}/main.py 9647 &
wait
