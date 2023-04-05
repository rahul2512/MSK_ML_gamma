#!/bin/bash -l
#SBATCH -J NN_79
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5616 &
python  ${path}/main.py 5617 &
python  ${path}/main.py 5618 &
python  ${path}/main.py 5619 &
python  ${path}/main.py 5620 &
python  ${path}/main.py 5621 &
python  ${path}/main.py 5622 &
python  ${path}/main.py 5623 &
python  ${path}/main.py 5624 &
python  ${path}/main.py 5625 &
python  ${path}/main.py 5626 &
python  ${path}/main.py 5627 &
python  ${path}/main.py 5628 &
python  ${path}/main.py 5629 &
python  ${path}/main.py 5630 &
python  ${path}/main.py 5631 &
python  ${path}/main.py 5632 &
python  ${path}/main.py 5633 &
python  ${path}/main.py 5634 &
python  ${path}/main.py 5635 &
python  ${path}/main.py 5636 &
python  ${path}/main.py 5637 &
python  ${path}/main.py 5638 &
python  ${path}/main.py 5639 &
python  ${path}/main.py 5640 &
python  ${path}/main.py 5641 &
python  ${path}/main.py 5642 &
python  ${path}/main.py 5643 &
python  ${path}/main.py 5644 &
python  ${path}/main.py 5645 &
python  ${path}/main.py 5646 &
python  ${path}/main.py 5647 &
python  ${path}/main.py 5648 &
python  ${path}/main.py 5649 &
python  ${path}/main.py 5650 &
python  ${path}/main.py 5651 &
python  ${path}/main.py 5652 &
python  ${path}/main.py 5653 &
python  ${path}/main.py 5654 &
python  ${path}/main.py 5655 &
python  ${path}/main.py 5656 &
python  ${path}/main.py 5657 &
python  ${path}/main.py 5658 &
python  ${path}/main.py 5659 &
python  ${path}/main.py 5660 &
python  ${path}/main.py 5661 &
python  ${path}/main.py 5662 &
python  ${path}/main.py 5663 &
python  ${path}/main.py 5664 &
python  ${path}/main.py 5665 &
python  ${path}/main.py 5666 &
python  ${path}/main.py 5667 &
python  ${path}/main.py 5668 &
python  ${path}/main.py 5669 &
python  ${path}/main.py 5670 &
python  ${path}/main.py 5671 &
python  ${path}/main.py 5672 &
python  ${path}/main.py 5673 &
python  ${path}/main.py 5674 &
python  ${path}/main.py 5675 &
python  ${path}/main.py 5676 &
python  ${path}/main.py 5677 &
python  ${path}/main.py 5678 &
python  ${path}/main.py 5679 &
python  ${path}/main.py 5680 &
python  ${path}/main.py 5681 &
python  ${path}/main.py 5682 &
python  ${path}/main.py 5683 &
python  ${path}/main.py 5684 &
python  ${path}/main.py 5685 &
python  ${path}/main.py 5686 &
python  ${path}/main.py 5687 &
wait
