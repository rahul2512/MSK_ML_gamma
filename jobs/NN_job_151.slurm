#!/bin/bash -l
#SBATCH -J NN_151
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 10800 &
python  ${path}/main.py 10801 &
python  ${path}/main.py 10802 &
python  ${path}/main.py 10803 &
python  ${path}/main.py 10804 &
python  ${path}/main.py 10805 &
python  ${path}/main.py 10806 &
python  ${path}/main.py 10807 &
python  ${path}/main.py 10808 &
python  ${path}/main.py 10809 &
python  ${path}/main.py 10810 &
python  ${path}/main.py 10811 &
python  ${path}/main.py 10812 &
python  ${path}/main.py 10813 &
python  ${path}/main.py 10814 &
python  ${path}/main.py 10815 &
python  ${path}/main.py 10816 &
python  ${path}/main.py 10817 &
python  ${path}/main.py 10818 &
python  ${path}/main.py 10819 &
python  ${path}/main.py 10820 &
python  ${path}/main.py 10821 &
python  ${path}/main.py 10822 &
python  ${path}/main.py 10823 &
python  ${path}/main.py 10824 &
python  ${path}/main.py 10825 &
python  ${path}/main.py 10826 &
python  ${path}/main.py 10827 &
python  ${path}/main.py 10828 &
python  ${path}/main.py 10829 &
python  ${path}/main.py 10830 &
python  ${path}/main.py 10831 &
python  ${path}/main.py 10832 &
python  ${path}/main.py 10833 &
python  ${path}/main.py 10834 &
python  ${path}/main.py 10835 &
python  ${path}/main.py 10836 &
python  ${path}/main.py 10837 &
python  ${path}/main.py 10838 &
python  ${path}/main.py 10839 &
python  ${path}/main.py 10840 &
python  ${path}/main.py 10841 &
python  ${path}/main.py 10842 &
python  ${path}/main.py 10843 &
python  ${path}/main.py 10844 &
python  ${path}/main.py 10845 &
python  ${path}/main.py 10846 &
python  ${path}/main.py 10847 &
python  ${path}/main.py 10848 &
python  ${path}/main.py 10849 &
python  ${path}/main.py 10850 &
python  ${path}/main.py 10851 &
python  ${path}/main.py 10852 &
python  ${path}/main.py 10853 &
python  ${path}/main.py 10854 &
python  ${path}/main.py 10855 &
python  ${path}/main.py 10856 &
python  ${path}/main.py 10857 &
python  ${path}/main.py 10858 &
python  ${path}/main.py 10859 &
python  ${path}/main.py 10860 &
python  ${path}/main.py 10861 &
python  ${path}/main.py 10862 &
python  ${path}/main.py 10863 &
python  ${path}/main.py 10864 &
python  ${path}/main.py 10865 &
python  ${path}/main.py 10866 &
python  ${path}/main.py 10867 &
python  ${path}/main.py 10868 &
python  ${path}/main.py 10869 &
python  ${path}/main.py 10870 &
python  ${path}/main.py 10871 &
wait
