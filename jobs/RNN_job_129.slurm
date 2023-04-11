#!/bin/bash -l
#SBATCH -J RNN_129
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4608 &
python  ${path}/main.py 4609 &
python  ${path}/main.py 4610 &
python  ${path}/main.py 4611 &
python  ${path}/main.py 4612 &
python  ${path}/main.py 4613 &
python  ${path}/main.py 4614 &
python  ${path}/main.py 4615 &
python  ${path}/main.py 4616 &
python  ${path}/main.py 4617 &
python  ${path}/main.py 4618 &
python  ${path}/main.py 4619 &
python  ${path}/main.py 4620 &
python  ${path}/main.py 4621 &
python  ${path}/main.py 4622 &
python  ${path}/main.py 4623 &
python  ${path}/main.py 4624 &
python  ${path}/main.py 4625 &
python  ${path}/main.py 4626 &
python  ${path}/main.py 4627 &
python  ${path}/main.py 4628 &
python  ${path}/main.py 4629 &
python  ${path}/main.py 4630 &
python  ${path}/main.py 4631 &
python  ${path}/main.py 4632 &
python  ${path}/main.py 4633 &
python  ${path}/main.py 4634 &
python  ${path}/main.py 4635 &
python  ${path}/main.py 4636 &
python  ${path}/main.py 4637 &
python  ${path}/main.py 4638 &
python  ${path}/main.py 4639 &
python  ${path}/main.py 4640 &
python  ${path}/main.py 4641 &
python  ${path}/main.py 4642 &
python  ${path}/main.py 4643 &
wait
