#!/bin/bash -l
#SBATCH -J NN_81
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5760 &
python  ${path}/main.py 5761 &
python  ${path}/main.py 5762 &
python  ${path}/main.py 5763 &
python  ${path}/main.py 5764 &
python  ${path}/main.py 5765 &
python  ${path}/main.py 5766 &
python  ${path}/main.py 5767 &
python  ${path}/main.py 5768 &
python  ${path}/main.py 5769 &
python  ${path}/main.py 5770 &
python  ${path}/main.py 5771 &
python  ${path}/main.py 5772 &
python  ${path}/main.py 5773 &
python  ${path}/main.py 5774 &
python  ${path}/main.py 5775 &
python  ${path}/main.py 5776 &
python  ${path}/main.py 5777 &
python  ${path}/main.py 5778 &
python  ${path}/main.py 5779 &
python  ${path}/main.py 5780 &
python  ${path}/main.py 5781 &
python  ${path}/main.py 5782 &
python  ${path}/main.py 5783 &
python  ${path}/main.py 5784 &
python  ${path}/main.py 5785 &
python  ${path}/main.py 5786 &
python  ${path}/main.py 5787 &
python  ${path}/main.py 5788 &
python  ${path}/main.py 5789 &
python  ${path}/main.py 5790 &
python  ${path}/main.py 5791 &
python  ${path}/main.py 5792 &
python  ${path}/main.py 5793 &
python  ${path}/main.py 5794 &
python  ${path}/main.py 5795 &
python  ${path}/main.py 5796 &
python  ${path}/main.py 5797 &
python  ${path}/main.py 5798 &
python  ${path}/main.py 5799 &
python  ${path}/main.py 5800 &
python  ${path}/main.py 5801 &
python  ${path}/main.py 5802 &
python  ${path}/main.py 5803 &
python  ${path}/main.py 5804 &
python  ${path}/main.py 5805 &
python  ${path}/main.py 5806 &
python  ${path}/main.py 5807 &
python  ${path}/main.py 5808 &
python  ${path}/main.py 5809 &
python  ${path}/main.py 5810 &
python  ${path}/main.py 5811 &
python  ${path}/main.py 5812 &
python  ${path}/main.py 5813 &
python  ${path}/main.py 5814 &
python  ${path}/main.py 5815 &
python  ${path}/main.py 5816 &
python  ${path}/main.py 5817 &
python  ${path}/main.py 5818 &
python  ${path}/main.py 5819 &
python  ${path}/main.py 5820 &
python  ${path}/main.py 5821 &
python  ${path}/main.py 5822 &
python  ${path}/main.py 5823 &
python  ${path}/main.py 5824 &
python  ${path}/main.py 5825 &
python  ${path}/main.py 5826 &
python  ${path}/main.py 5827 &
python  ${path}/main.py 5828 &
python  ${path}/main.py 5829 &
python  ${path}/main.py 5830 &
python  ${path}/main.py 5831 &
wait
