#!/bin/bash -l
#SBATCH -J RNN_101
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3600 &
python  ${path}/main.py 3601 &
python  ${path}/main.py 3602 &
python  ${path}/main.py 3603 &
python  ${path}/main.py 3604 &
python  ${path}/main.py 3605 &
python  ${path}/main.py 3606 &
python  ${path}/main.py 3607 &
python  ${path}/main.py 3608 &
python  ${path}/main.py 3609 &
python  ${path}/main.py 3610 &
python  ${path}/main.py 3611 &
python  ${path}/main.py 3612 &
python  ${path}/main.py 3613 &
python  ${path}/main.py 3614 &
python  ${path}/main.py 3615 &
python  ${path}/main.py 3616 &
python  ${path}/main.py 3617 &
python  ${path}/main.py 3618 &
python  ${path}/main.py 3619 &
python  ${path}/main.py 3620 &
python  ${path}/main.py 3621 &
python  ${path}/main.py 3622 &
python  ${path}/main.py 3623 &
python  ${path}/main.py 3624 &
python  ${path}/main.py 3625 &
python  ${path}/main.py 3626 &
python  ${path}/main.py 3627 &
python  ${path}/main.py 3628 &
python  ${path}/main.py 3629 &
python  ${path}/main.py 3630 &
python  ${path}/main.py 3631 &
python  ${path}/main.py 3632 &
python  ${path}/main.py 3633 &
python  ${path}/main.py 3634 &
python  ${path}/main.py 3635 &
wait
