#!/bin/bash -l
#SBATCH -J RNN_209
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7488 &
python  ${path}/main.py 7489 &
python  ${path}/main.py 7490 &
python  ${path}/main.py 7491 &
python  ${path}/main.py 7492 &
python  ${path}/main.py 7493 &
python  ${path}/main.py 7494 &
python  ${path}/main.py 7495 &
python  ${path}/main.py 7496 &
python  ${path}/main.py 7497 &
python  ${path}/main.py 7498 &
python  ${path}/main.py 7499 &
python  ${path}/main.py 7500 &
python  ${path}/main.py 7501 &
python  ${path}/main.py 7502 &
python  ${path}/main.py 7503 &
python  ${path}/main.py 7504 &
python  ${path}/main.py 7505 &
python  ${path}/main.py 7506 &
python  ${path}/main.py 7507 &
python  ${path}/main.py 7508 &
python  ${path}/main.py 7509 &
python  ${path}/main.py 7510 &
python  ${path}/main.py 7511 &
python  ${path}/main.py 7512 &
python  ${path}/main.py 7513 &
python  ${path}/main.py 7514 &
python  ${path}/main.py 7515 &
python  ${path}/main.py 7516 &
python  ${path}/main.py 7517 &
python  ${path}/main.py 7518 &
python  ${path}/main.py 7519 &
python  ${path}/main.py 7520 &
python  ${path}/main.py 7521 &
python  ${path}/main.py 7522 &
python  ${path}/main.py 7523 &
wait
