#!/bin/bash -l
#SBATCH -J NN_8
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 504 &
python  ${path}/main.py 505 &
python  ${path}/main.py 506 &
python  ${path}/main.py 507 &
python  ${path}/main.py 508 &
python  ${path}/main.py 509 &
python  ${path}/main.py 510 &
python  ${path}/main.py 511 &
python  ${path}/main.py 512 &
python  ${path}/main.py 513 &
python  ${path}/main.py 514 &
python  ${path}/main.py 515 &
python  ${path}/main.py 516 &
python  ${path}/main.py 517 &
python  ${path}/main.py 518 &
python  ${path}/main.py 519 &
python  ${path}/main.py 520 &
python  ${path}/main.py 521 &
python  ${path}/main.py 522 &
python  ${path}/main.py 523 &
python  ${path}/main.py 524 &
python  ${path}/main.py 525 &
python  ${path}/main.py 526 &
python  ${path}/main.py 527 &
python  ${path}/main.py 528 &
python  ${path}/main.py 529 &
python  ${path}/main.py 530 &
python  ${path}/main.py 531 &
python  ${path}/main.py 532 &
python  ${path}/main.py 533 &
python  ${path}/main.py 534 &
python  ${path}/main.py 535 &
python  ${path}/main.py 536 &
python  ${path}/main.py 537 &
python  ${path}/main.py 538 &
python  ${path}/main.py 539 &
python  ${path}/main.py 540 &
python  ${path}/main.py 541 &
python  ${path}/main.py 542 &
python  ${path}/main.py 543 &
python  ${path}/main.py 544 &
python  ${path}/main.py 545 &
python  ${path}/main.py 546 &
python  ${path}/main.py 547 &
python  ${path}/main.py 548 &
python  ${path}/main.py 549 &
python  ${path}/main.py 550 &
python  ${path}/main.py 551 &
python  ${path}/main.py 552 &
python  ${path}/main.py 553 &
python  ${path}/main.py 554 &
python  ${path}/main.py 555 &
python  ${path}/main.py 556 &
python  ${path}/main.py 557 &
python  ${path}/main.py 558 &
python  ${path}/main.py 559 &
python  ${path}/main.py 560 &
python  ${path}/main.py 561 &
python  ${path}/main.py 562 &
python  ${path}/main.py 563 &
python  ${path}/main.py 564 &
python  ${path}/main.py 565 &
python  ${path}/main.py 566 &
python  ${path}/main.py 567 &
python  ${path}/main.py 568 &
python  ${path}/main.py 569 &
python  ${path}/main.py 570 &
python  ${path}/main.py 571 &
python  ${path}/main.py 572 &
python  ${path}/main.py 573 &
python  ${path}/main.py 574 &
python  ${path}/main.py 575 &
wait
