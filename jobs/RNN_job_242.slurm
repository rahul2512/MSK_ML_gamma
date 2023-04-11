#!/bin/bash -l
#SBATCH -J RNN_242
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
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
