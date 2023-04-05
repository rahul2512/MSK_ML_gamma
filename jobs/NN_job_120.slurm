#!/bin/bash -l
#SBATCH -J NN_120
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8568 &
python  ${path}/main.py 8569 &
python  ${path}/main.py 8570 &
python  ${path}/main.py 8571 &
python  ${path}/main.py 8572 &
python  ${path}/main.py 8573 &
python  ${path}/main.py 8574 &
python  ${path}/main.py 8575 &
python  ${path}/main.py 8576 &
python  ${path}/main.py 8577 &
python  ${path}/main.py 8578 &
python  ${path}/main.py 8579 &
python  ${path}/main.py 8580 &
python  ${path}/main.py 8581 &
python  ${path}/main.py 8582 &
python  ${path}/main.py 8583 &
python  ${path}/main.py 8584 &
python  ${path}/main.py 8585 &
python  ${path}/main.py 8586 &
python  ${path}/main.py 8587 &
python  ${path}/main.py 8588 &
python  ${path}/main.py 8589 &
python  ${path}/main.py 8590 &
python  ${path}/main.py 8591 &
python  ${path}/main.py 8592 &
python  ${path}/main.py 8593 &
python  ${path}/main.py 8594 &
python  ${path}/main.py 8595 &
python  ${path}/main.py 8596 &
python  ${path}/main.py 8597 &
python  ${path}/main.py 8598 &
python  ${path}/main.py 8599 &
python  ${path}/main.py 8600 &
python  ${path}/main.py 8601 &
python  ${path}/main.py 8602 &
python  ${path}/main.py 8603 &
python  ${path}/main.py 8604 &
python  ${path}/main.py 8605 &
python  ${path}/main.py 8606 &
python  ${path}/main.py 8607 &
python  ${path}/main.py 8608 &
python  ${path}/main.py 8609 &
python  ${path}/main.py 8610 &
python  ${path}/main.py 8611 &
python  ${path}/main.py 8612 &
python  ${path}/main.py 8613 &
python  ${path}/main.py 8614 &
python  ${path}/main.py 8615 &
python  ${path}/main.py 8616 &
python  ${path}/main.py 8617 &
python  ${path}/main.py 8618 &
python  ${path}/main.py 8619 &
python  ${path}/main.py 8620 &
python  ${path}/main.py 8621 &
python  ${path}/main.py 8622 &
python  ${path}/main.py 8623 &
python  ${path}/main.py 8624 &
python  ${path}/main.py 8625 &
python  ${path}/main.py 8626 &
python  ${path}/main.py 8627 &
python  ${path}/main.py 8628 &
python  ${path}/main.py 8629 &
python  ${path}/main.py 8630 &
python  ${path}/main.py 8631 &
python  ${path}/main.py 8632 &
python  ${path}/main.py 8633 &
python  ${path}/main.py 8634 &
python  ${path}/main.py 8635 &
python  ${path}/main.py 8636 &
python  ${path}/main.py 8637 &
python  ${path}/main.py 8638 &
python  ${path}/main.py 8639 &
wait
