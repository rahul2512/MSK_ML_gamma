#!/bin/bash -l
#SBATCH -J RNN_131
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4680 &
python  ${path}/main.py 4681 &
python  ${path}/main.py 4682 &
python  ${path}/main.py 4683 &
python  ${path}/main.py 4684 &
python  ${path}/main.py 4685 &
python  ${path}/main.py 4686 &
python  ${path}/main.py 4687 &
python  ${path}/main.py 4688 &
python  ${path}/main.py 4689 &
python  ${path}/main.py 4690 &
python  ${path}/main.py 4691 &
python  ${path}/main.py 4692 &
python  ${path}/main.py 4693 &
python  ${path}/main.py 4694 &
python  ${path}/main.py 4695 &
python  ${path}/main.py 4696 &
python  ${path}/main.py 4697 &
python  ${path}/main.py 4698 &
python  ${path}/main.py 4699 &
python  ${path}/main.py 4700 &
python  ${path}/main.py 4701 &
python  ${path}/main.py 4702 &
python  ${path}/main.py 4703 &
python  ${path}/main.py 4704 &
python  ${path}/main.py 4705 &
python  ${path}/main.py 4706 &
python  ${path}/main.py 4707 &
python  ${path}/main.py 4708 &
python  ${path}/main.py 4709 &
python  ${path}/main.py 4710 &
python  ${path}/main.py 4711 &
python  ${path}/main.py 4712 &
python  ${path}/main.py 4713 &
python  ${path}/main.py 4714 &
python  ${path}/main.py 4715 &
wait
