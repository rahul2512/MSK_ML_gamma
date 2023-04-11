#!/bin/bash -l
#SBATCH -J RNN_208
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7452 &
python  ${path}/main.py 7453 &
python  ${path}/main.py 7454 &
python  ${path}/main.py 7455 &
python  ${path}/main.py 7456 &
python  ${path}/main.py 7457 &
python  ${path}/main.py 7458 &
python  ${path}/main.py 7459 &
python  ${path}/main.py 7460 &
python  ${path}/main.py 7461 &
python  ${path}/main.py 7462 &
python  ${path}/main.py 7463 &
python  ${path}/main.py 7464 &
python  ${path}/main.py 7465 &
python  ${path}/main.py 7466 &
python  ${path}/main.py 7467 &
python  ${path}/main.py 7468 &
python  ${path}/main.py 7469 &
python  ${path}/main.py 7470 &
python  ${path}/main.py 7471 &
python  ${path}/main.py 7472 &
python  ${path}/main.py 7473 &
python  ${path}/main.py 7474 &
python  ${path}/main.py 7475 &
python  ${path}/main.py 7476 &
python  ${path}/main.py 7477 &
python  ${path}/main.py 7478 &
python  ${path}/main.py 7479 &
python  ${path}/main.py 7480 &
python  ${path}/main.py 7481 &
python  ${path}/main.py 7482 &
python  ${path}/main.py 7483 &
python  ${path}/main.py 7484 &
python  ${path}/main.py 7485 &
python  ${path}/main.py 7486 &
python  ${path}/main.py 7487 &
wait
