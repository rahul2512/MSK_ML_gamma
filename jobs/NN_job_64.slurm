#!/bin/bash -l
#SBATCH -J NN_64
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4536 &
python  ${path}/main.py 4537 &
python  ${path}/main.py 4538 &
python  ${path}/main.py 4539 &
python  ${path}/main.py 4540 &
python  ${path}/main.py 4541 &
python  ${path}/main.py 4542 &
python  ${path}/main.py 4543 &
python  ${path}/main.py 4544 &
python  ${path}/main.py 4545 &
python  ${path}/main.py 4546 &
python  ${path}/main.py 4547 &
python  ${path}/main.py 4548 &
python  ${path}/main.py 4549 &
python  ${path}/main.py 4550 &
python  ${path}/main.py 4551 &
python  ${path}/main.py 4552 &
python  ${path}/main.py 4553 &
python  ${path}/main.py 4554 &
python  ${path}/main.py 4555 &
python  ${path}/main.py 4556 &
python  ${path}/main.py 4557 &
python  ${path}/main.py 4558 &
python  ${path}/main.py 4559 &
python  ${path}/main.py 4560 &
python  ${path}/main.py 4561 &
python  ${path}/main.py 4562 &
python  ${path}/main.py 4563 &
python  ${path}/main.py 4564 &
python  ${path}/main.py 4565 &
python  ${path}/main.py 4566 &
python  ${path}/main.py 4567 &
python  ${path}/main.py 4568 &
python  ${path}/main.py 4569 &
python  ${path}/main.py 4570 &
python  ${path}/main.py 4571 &
python  ${path}/main.py 4572 &
python  ${path}/main.py 4573 &
python  ${path}/main.py 4574 &
python  ${path}/main.py 4575 &
python  ${path}/main.py 4576 &
python  ${path}/main.py 4577 &
python  ${path}/main.py 4578 &
python  ${path}/main.py 4579 &
python  ${path}/main.py 4580 &
python  ${path}/main.py 4581 &
python  ${path}/main.py 4582 &
python  ${path}/main.py 4583 &
python  ${path}/main.py 4584 &
python  ${path}/main.py 4585 &
python  ${path}/main.py 4586 &
python  ${path}/main.py 4587 &
python  ${path}/main.py 4588 &
python  ${path}/main.py 4589 &
python  ${path}/main.py 4590 &
python  ${path}/main.py 4591 &
python  ${path}/main.py 4592 &
python  ${path}/main.py 4593 &
python  ${path}/main.py 4594 &
python  ${path}/main.py 4595 &
python  ${path}/main.py 4596 &
python  ${path}/main.py 4597 &
python  ${path}/main.py 4598 &
python  ${path}/main.py 4599 &
python  ${path}/main.py 4600 &
python  ${path}/main.py 4601 &
python  ${path}/main.py 4602 &
python  ${path}/main.py 4603 &
python  ${path}/main.py 4604 &
python  ${path}/main.py 4605 &
python  ${path}/main.py 4606 &
python  ${path}/main.py 4607 &
wait
