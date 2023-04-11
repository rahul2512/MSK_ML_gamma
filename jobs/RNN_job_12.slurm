#!/bin/bash -l
#SBATCH -J RNN_12
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 396 &
python  ${path}/main.py 397 &
python  ${path}/main.py 398 &
python  ${path}/main.py 399 &
python  ${path}/main.py 400 &
python  ${path}/main.py 401 &
python  ${path}/main.py 402 &
python  ${path}/main.py 403 &
python  ${path}/main.py 404 &
python  ${path}/main.py 405 &
python  ${path}/main.py 406 &
python  ${path}/main.py 407 &
python  ${path}/main.py 408 &
python  ${path}/main.py 409 &
python  ${path}/main.py 410 &
python  ${path}/main.py 411 &
python  ${path}/main.py 412 &
python  ${path}/main.py 413 &
python  ${path}/main.py 414 &
python  ${path}/main.py 415 &
python  ${path}/main.py 416 &
python  ${path}/main.py 417 &
python  ${path}/main.py 418 &
python  ${path}/main.py 419 &
python  ${path}/main.py 420 &
python  ${path}/main.py 421 &
python  ${path}/main.py 422 &
python  ${path}/main.py 423 &
python  ${path}/main.py 424 &
python  ${path}/main.py 425 &
python  ${path}/main.py 426 &
python  ${path}/main.py 427 &
python  ${path}/main.py 428 &
python  ${path}/main.py 429 &
python  ${path}/main.py 430 &
python  ${path}/main.py 431 &
wait
