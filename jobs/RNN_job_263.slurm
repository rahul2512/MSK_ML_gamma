#!/bin/bash -l
#SBATCH -J RNN_263
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9432 &
python  ${path}/main.py 9433 &
python  ${path}/main.py 9434 &
python  ${path}/main.py 9435 &
python  ${path}/main.py 9436 &
python  ${path}/main.py 9437 &
python  ${path}/main.py 9438 &
python  ${path}/main.py 9439 &
python  ${path}/main.py 9440 &
python  ${path}/main.py 9441 &
python  ${path}/main.py 9442 &
python  ${path}/main.py 9443 &
python  ${path}/main.py 9444 &
python  ${path}/main.py 9445 &
python  ${path}/main.py 9446 &
python  ${path}/main.py 9447 &
python  ${path}/main.py 9448 &
python  ${path}/main.py 9449 &
python  ${path}/main.py 9450 &
python  ${path}/main.py 9451 &
python  ${path}/main.py 9452 &
python  ${path}/main.py 9453 &
python  ${path}/main.py 9454 &
python  ${path}/main.py 9455 &
python  ${path}/main.py 9456 &
python  ${path}/main.py 9457 &
python  ${path}/main.py 9458 &
python  ${path}/main.py 9459 &
python  ${path}/main.py 9460 &
python  ${path}/main.py 9461 &
python  ${path}/main.py 9462 &
python  ${path}/main.py 9463 &
python  ${path}/main.py 9464 &
python  ${path}/main.py 9465 &
python  ${path}/main.py 9466 &
python  ${path}/main.py 9467 &
wait
