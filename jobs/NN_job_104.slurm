#!/bin/bash -l
#SBATCH -J NN_104
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7416 &
python  ${path}/main.py 7417 &
python  ${path}/main.py 7418 &
python  ${path}/main.py 7419 &
python  ${path}/main.py 7420 &
python  ${path}/main.py 7421 &
python  ${path}/main.py 7422 &
python  ${path}/main.py 7423 &
python  ${path}/main.py 7424 &
python  ${path}/main.py 7425 &
python  ${path}/main.py 7426 &
python  ${path}/main.py 7427 &
python  ${path}/main.py 7428 &
python  ${path}/main.py 7429 &
python  ${path}/main.py 7430 &
python  ${path}/main.py 7431 &
python  ${path}/main.py 7432 &
python  ${path}/main.py 7433 &
python  ${path}/main.py 7434 &
python  ${path}/main.py 7435 &
python  ${path}/main.py 7436 &
python  ${path}/main.py 7437 &
python  ${path}/main.py 7438 &
python  ${path}/main.py 7439 &
python  ${path}/main.py 7440 &
python  ${path}/main.py 7441 &
python  ${path}/main.py 7442 &
python  ${path}/main.py 7443 &
python  ${path}/main.py 7444 &
python  ${path}/main.py 7445 &
python  ${path}/main.py 7446 &
python  ${path}/main.py 7447 &
python  ${path}/main.py 7448 &
python  ${path}/main.py 7449 &
python  ${path}/main.py 7450 &
python  ${path}/main.py 7451 &
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
