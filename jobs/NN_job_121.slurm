#!/bin/bash -l
#SBATCH -J NN_121
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 8640 &
python  ${path}/main.py 8641 &
python  ${path}/main.py 8642 &
python  ${path}/main.py 8643 &
python  ${path}/main.py 8644 &
python  ${path}/main.py 8645 &
python  ${path}/main.py 8646 &
python  ${path}/main.py 8647 &
python  ${path}/main.py 8648 &
python  ${path}/main.py 8649 &
python  ${path}/main.py 8650 &
python  ${path}/main.py 8651 &
python  ${path}/main.py 8652 &
python  ${path}/main.py 8653 &
python  ${path}/main.py 8654 &
python  ${path}/main.py 8655 &
python  ${path}/main.py 8656 &
python  ${path}/main.py 8657 &
python  ${path}/main.py 8658 &
python  ${path}/main.py 8659 &
python  ${path}/main.py 8660 &
python  ${path}/main.py 8661 &
python  ${path}/main.py 8662 &
python  ${path}/main.py 8663 &
python  ${path}/main.py 8664 &
python  ${path}/main.py 8665 &
python  ${path}/main.py 8666 &
python  ${path}/main.py 8667 &
python  ${path}/main.py 8668 &
python  ${path}/main.py 8669 &
python  ${path}/main.py 8670 &
python  ${path}/main.py 8671 &
python  ${path}/main.py 8672 &
python  ${path}/main.py 8673 &
python  ${path}/main.py 8674 &
python  ${path}/main.py 8675 &
python  ${path}/main.py 8676 &
python  ${path}/main.py 8677 &
python  ${path}/main.py 8678 &
python  ${path}/main.py 8679 &
python  ${path}/main.py 8680 &
python  ${path}/main.py 8681 &
python  ${path}/main.py 8682 &
python  ${path}/main.py 8683 &
python  ${path}/main.py 8684 &
python  ${path}/main.py 8685 &
python  ${path}/main.py 8686 &
python  ${path}/main.py 8687 &
python  ${path}/main.py 8688 &
python  ${path}/main.py 8689 &
python  ${path}/main.py 8690 &
python  ${path}/main.py 8691 &
python  ${path}/main.py 8692 &
python  ${path}/main.py 8693 &
python  ${path}/main.py 8694 &
python  ${path}/main.py 8695 &
python  ${path}/main.py 8696 &
python  ${path}/main.py 8697 &
python  ${path}/main.py 8698 &
python  ${path}/main.py 8699 &
python  ${path}/main.py 8700 &
python  ${path}/main.py 8701 &
python  ${path}/main.py 8702 &
python  ${path}/main.py 8703 &
python  ${path}/main.py 8704 &
python  ${path}/main.py 8705 &
python  ${path}/main.py 8706 &
python  ${path}/main.py 8707 &
python  ${path}/main.py 8708 &
python  ${path}/main.py 8709 &
python  ${path}/main.py 8710 &
python  ${path}/main.py 8711 &
wait
