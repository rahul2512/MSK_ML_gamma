#!/bin/bash -l
#SBATCH -J RNN_124
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4428 &
python  ${path}/main.py 4429 &
python  ${path}/main.py 4430 &
python  ${path}/main.py 4431 &
python  ${path}/main.py 4432 &
python  ${path}/main.py 4433 &
python  ${path}/main.py 4434 &
python  ${path}/main.py 4435 &
python  ${path}/main.py 4436 &
python  ${path}/main.py 4437 &
python  ${path}/main.py 4438 &
python  ${path}/main.py 4439 &
python  ${path}/main.py 4440 &
python  ${path}/main.py 4441 &
python  ${path}/main.py 4442 &
python  ${path}/main.py 4443 &
python  ${path}/main.py 4444 &
python  ${path}/main.py 4445 &
python  ${path}/main.py 4446 &
python  ${path}/main.py 4447 &
python  ${path}/main.py 4448 &
python  ${path}/main.py 4449 &
python  ${path}/main.py 4450 &
python  ${path}/main.py 4451 &
python  ${path}/main.py 4452 &
python  ${path}/main.py 4453 &
python  ${path}/main.py 4454 &
python  ${path}/main.py 4455 &
python  ${path}/main.py 4456 &
python  ${path}/main.py 4457 &
python  ${path}/main.py 4458 &
python  ${path}/main.py 4459 &
python  ${path}/main.py 4460 &
python  ${path}/main.py 4461 &
python  ${path}/main.py 4462 &
python  ${path}/main.py 4463 &
wait
