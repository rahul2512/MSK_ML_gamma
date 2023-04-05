#!/bin/bash -l
#SBATCH -J NN_109
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7776 &
python  ${path}/main.py 7777 &
python  ${path}/main.py 7778 &
python  ${path}/main.py 7779 &
python  ${path}/main.py 7780 &
python  ${path}/main.py 7781 &
python  ${path}/main.py 7782 &
python  ${path}/main.py 7783 &
python  ${path}/main.py 7784 &
python  ${path}/main.py 7785 &
python  ${path}/main.py 7786 &
python  ${path}/main.py 7787 &
python  ${path}/main.py 7788 &
python  ${path}/main.py 7789 &
python  ${path}/main.py 7790 &
python  ${path}/main.py 7791 &
python  ${path}/main.py 7792 &
python  ${path}/main.py 7793 &
python  ${path}/main.py 7794 &
python  ${path}/main.py 7795 &
python  ${path}/main.py 7796 &
python  ${path}/main.py 7797 &
python  ${path}/main.py 7798 &
python  ${path}/main.py 7799 &
python  ${path}/main.py 7800 &
python  ${path}/main.py 7801 &
python  ${path}/main.py 7802 &
python  ${path}/main.py 7803 &
python  ${path}/main.py 7804 &
python  ${path}/main.py 7805 &
python  ${path}/main.py 7806 &
python  ${path}/main.py 7807 &
python  ${path}/main.py 7808 &
python  ${path}/main.py 7809 &
python  ${path}/main.py 7810 &
python  ${path}/main.py 7811 &
python  ${path}/main.py 7812 &
python  ${path}/main.py 7813 &
python  ${path}/main.py 7814 &
python  ${path}/main.py 7815 &
python  ${path}/main.py 7816 &
python  ${path}/main.py 7817 &
python  ${path}/main.py 7818 &
python  ${path}/main.py 7819 &
python  ${path}/main.py 7820 &
python  ${path}/main.py 7821 &
python  ${path}/main.py 7822 &
python  ${path}/main.py 7823 &
python  ${path}/main.py 7824 &
python  ${path}/main.py 7825 &
python  ${path}/main.py 7826 &
python  ${path}/main.py 7827 &
python  ${path}/main.py 7828 &
python  ${path}/main.py 7829 &
python  ${path}/main.py 7830 &
python  ${path}/main.py 7831 &
python  ${path}/main.py 7832 &
python  ${path}/main.py 7833 &
python  ${path}/main.py 7834 &
python  ${path}/main.py 7835 &
python  ${path}/main.py 7836 &
python  ${path}/main.py 7837 &
python  ${path}/main.py 7838 &
python  ${path}/main.py 7839 &
python  ${path}/main.py 7840 &
python  ${path}/main.py 7841 &
python  ${path}/main.py 7842 &
python  ${path}/main.py 7843 &
python  ${path}/main.py 7844 &
python  ${path}/main.py 7845 &
python  ${path}/main.py 7846 &
python  ${path}/main.py 7847 &
wait
