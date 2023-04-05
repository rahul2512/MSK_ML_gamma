#!/bin/bash -l
#SBATCH -J NN_54
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 3816 &
python  ${path}/main.py 3817 &
python  ${path}/main.py 3818 &
python  ${path}/main.py 3819 &
python  ${path}/main.py 3820 &
python  ${path}/main.py 3821 &
python  ${path}/main.py 3822 &
python  ${path}/main.py 3823 &
python  ${path}/main.py 3824 &
python  ${path}/main.py 3825 &
python  ${path}/main.py 3826 &
python  ${path}/main.py 3827 &
python  ${path}/main.py 3828 &
python  ${path}/main.py 3829 &
python  ${path}/main.py 3830 &
python  ${path}/main.py 3831 &
python  ${path}/main.py 3832 &
python  ${path}/main.py 3833 &
python  ${path}/main.py 3834 &
python  ${path}/main.py 3835 &
python  ${path}/main.py 3836 &
python  ${path}/main.py 3837 &
python  ${path}/main.py 3838 &
python  ${path}/main.py 3839 &
python  ${path}/main.py 3840 &
python  ${path}/main.py 3841 &
python  ${path}/main.py 3842 &
python  ${path}/main.py 3843 &
python  ${path}/main.py 3844 &
python  ${path}/main.py 3845 &
python  ${path}/main.py 3846 &
python  ${path}/main.py 3847 &
python  ${path}/main.py 3848 &
python  ${path}/main.py 3849 &
python  ${path}/main.py 3850 &
python  ${path}/main.py 3851 &
python  ${path}/main.py 3852 &
python  ${path}/main.py 3853 &
python  ${path}/main.py 3854 &
python  ${path}/main.py 3855 &
python  ${path}/main.py 3856 &
python  ${path}/main.py 3857 &
python  ${path}/main.py 3858 &
python  ${path}/main.py 3859 &
python  ${path}/main.py 3860 &
python  ${path}/main.py 3861 &
python  ${path}/main.py 3862 &
python  ${path}/main.py 3863 &
python  ${path}/main.py 3864 &
python  ${path}/main.py 3865 &
python  ${path}/main.py 3866 &
python  ${path}/main.py 3867 &
python  ${path}/main.py 3868 &
python  ${path}/main.py 3869 &
python  ${path}/main.py 3870 &
python  ${path}/main.py 3871 &
python  ${path}/main.py 3872 &
python  ${path}/main.py 3873 &
python  ${path}/main.py 3874 &
python  ${path}/main.py 3875 &
python  ${path}/main.py 3876 &
python  ${path}/main.py 3877 &
python  ${path}/main.py 3878 &
python  ${path}/main.py 3879 &
python  ${path}/main.py 3880 &
python  ${path}/main.py 3881 &
python  ${path}/main.py 3882 &
python  ${path}/main.py 3883 &
python  ${path}/main.py 3884 &
python  ${path}/main.py 3885 &
python  ${path}/main.py 3886 &
python  ${path}/main.py 3887 &
wait
