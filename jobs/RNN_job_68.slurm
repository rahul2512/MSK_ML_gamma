#!/bin/bash -l
#SBATCH -J RNN_68
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 2412 &
python  ${path}/main.py 2413 &
python  ${path}/main.py 2414 &
python  ${path}/main.py 2415 &
python  ${path}/main.py 2416 &
python  ${path}/main.py 2417 &
python  ${path}/main.py 2418 &
python  ${path}/main.py 2419 &
python  ${path}/main.py 2420 &
python  ${path}/main.py 2421 &
python  ${path}/main.py 2422 &
python  ${path}/main.py 2423 &
python  ${path}/main.py 2424 &
python  ${path}/main.py 2425 &
python  ${path}/main.py 2426 &
python  ${path}/main.py 2427 &
python  ${path}/main.py 2428 &
python  ${path}/main.py 2429 &
python  ${path}/main.py 2430 &
python  ${path}/main.py 2431 &
python  ${path}/main.py 2432 &
python  ${path}/main.py 2433 &
python  ${path}/main.py 2434 &
python  ${path}/main.py 2435 &
python  ${path}/main.py 2436 &
python  ${path}/main.py 2437 &
python  ${path}/main.py 2438 &
python  ${path}/main.py 2439 &
python  ${path}/main.py 2440 &
python  ${path}/main.py 2441 &
python  ${path}/main.py 2442 &
python  ${path}/main.py 2443 &
python  ${path}/main.py 2444 &
python  ${path}/main.py 2445 &
python  ${path}/main.py 2446 &
python  ${path}/main.py 2447 &
wait
