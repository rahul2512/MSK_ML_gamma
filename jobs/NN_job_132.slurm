#!/bin/bash -l
#SBATCH -J NN_132
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
python  ${path}/main.py 9468 &
python  ${path}/main.py 9469 &
python  ${path}/main.py 9470 &
python  ${path}/main.py 9471 &
python  ${path}/main.py 9472 &
python  ${path}/main.py 9473 &
python  ${path}/main.py 9474 &
python  ${path}/main.py 9475 &
python  ${path}/main.py 9476 &
python  ${path}/main.py 9477 &
python  ${path}/main.py 9478 &
python  ${path}/main.py 9479 &
python  ${path}/main.py 9480 &
python  ${path}/main.py 9481 &
python  ${path}/main.py 9482 &
python  ${path}/main.py 9483 &
python  ${path}/main.py 9484 &
python  ${path}/main.py 9485 &
python  ${path}/main.py 9486 &
python  ${path}/main.py 9487 &
python  ${path}/main.py 9488 &
python  ${path}/main.py 9489 &
python  ${path}/main.py 9490 &
python  ${path}/main.py 9491 &
python  ${path}/main.py 9492 &
python  ${path}/main.py 9493 &
python  ${path}/main.py 9494 &
python  ${path}/main.py 9495 &
python  ${path}/main.py 9496 &
python  ${path}/main.py 9497 &
python  ${path}/main.py 9498 &
python  ${path}/main.py 9499 &
python  ${path}/main.py 9500 &
python  ${path}/main.py 9501 &
python  ${path}/main.py 9502 &
python  ${path}/main.py 9503 &
wait
