#!/bin/bash -l
#SBATCH -J NN_49
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3456 &
python  ${path}/main.py 3457 &
python  ${path}/main.py 3458 &
python  ${path}/main.py 3459 &
python  ${path}/main.py 3460 &
python  ${path}/main.py 3461 &
python  ${path}/main.py 3462 &
python  ${path}/main.py 3463 &
python  ${path}/main.py 3464 &
python  ${path}/main.py 3465 &
python  ${path}/main.py 3466 &
python  ${path}/main.py 3467 &
python  ${path}/main.py 3468 &
python  ${path}/main.py 3469 &
python  ${path}/main.py 3470 &
python  ${path}/main.py 3471 &
python  ${path}/main.py 3472 &
python  ${path}/main.py 3473 &
python  ${path}/main.py 3474 &
python  ${path}/main.py 3475 &
python  ${path}/main.py 3476 &
python  ${path}/main.py 3477 &
python  ${path}/main.py 3478 &
python  ${path}/main.py 3479 &
python  ${path}/main.py 3480 &
python  ${path}/main.py 3481 &
python  ${path}/main.py 3482 &
python  ${path}/main.py 3483 &
python  ${path}/main.py 3484 &
python  ${path}/main.py 3485 &
python  ${path}/main.py 3486 &
python  ${path}/main.py 3487 &
python  ${path}/main.py 3488 &
python  ${path}/main.py 3489 &
python  ${path}/main.py 3490 &
python  ${path}/main.py 3491 &
python  ${path}/main.py 3492 &
python  ${path}/main.py 3493 &
python  ${path}/main.py 3494 &
python  ${path}/main.py 3495 &
python  ${path}/main.py 3496 &
python  ${path}/main.py 3497 &
python  ${path}/main.py 3498 &
python  ${path}/main.py 3499 &
python  ${path}/main.py 3500 &
python  ${path}/main.py 3501 &
python  ${path}/main.py 3502 &
python  ${path}/main.py 3503 &
python  ${path}/main.py 3504 &
python  ${path}/main.py 3505 &
python  ${path}/main.py 3506 &
python  ${path}/main.py 3507 &
python  ${path}/main.py 3508 &
python  ${path}/main.py 3509 &
python  ${path}/main.py 3510 &
python  ${path}/main.py 3511 &
python  ${path}/main.py 3512 &
python  ${path}/main.py 3513 &
python  ${path}/main.py 3514 &
python  ${path}/main.py 3515 &
python  ${path}/main.py 3516 &
python  ${path}/main.py 3517 &
python  ${path}/main.py 3518 &
python  ${path}/main.py 3519 &
python  ${path}/main.py 3520 &
python  ${path}/main.py 3521 &
python  ${path}/main.py 3522 &
python  ${path}/main.py 3523 &
python  ${path}/main.py 3524 &
python  ${path}/main.py 3525 &
python  ${path}/main.py 3526 &
python  ${path}/main.py 3527 &
wait
