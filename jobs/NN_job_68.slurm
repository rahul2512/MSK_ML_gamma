#!/bin/bash -l
#SBATCH -J NN_68
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4824 &
python  ${path}/main.py 4825 &
python  ${path}/main.py 4826 &
python  ${path}/main.py 4827 &
python  ${path}/main.py 4828 &
python  ${path}/main.py 4829 &
python  ${path}/main.py 4830 &
python  ${path}/main.py 4831 &
python  ${path}/main.py 4832 &
python  ${path}/main.py 4833 &
python  ${path}/main.py 4834 &
python  ${path}/main.py 4835 &
python  ${path}/main.py 4836 &
python  ${path}/main.py 4837 &
python  ${path}/main.py 4838 &
python  ${path}/main.py 4839 &
python  ${path}/main.py 4840 &
python  ${path}/main.py 4841 &
python  ${path}/main.py 4842 &
python  ${path}/main.py 4843 &
python  ${path}/main.py 4844 &
python  ${path}/main.py 4845 &
python  ${path}/main.py 4846 &
python  ${path}/main.py 4847 &
python  ${path}/main.py 4848 &
python  ${path}/main.py 4849 &
python  ${path}/main.py 4850 &
python  ${path}/main.py 4851 &
python  ${path}/main.py 4852 &
python  ${path}/main.py 4853 &
python  ${path}/main.py 4854 &
python  ${path}/main.py 4855 &
python  ${path}/main.py 4856 &
python  ${path}/main.py 4857 &
python  ${path}/main.py 4858 &
python  ${path}/main.py 4859 &
python  ${path}/main.py 4860 &
python  ${path}/main.py 4861 &
python  ${path}/main.py 4862 &
python  ${path}/main.py 4863 &
python  ${path}/main.py 4864 &
python  ${path}/main.py 4865 &
python  ${path}/main.py 4866 &
python  ${path}/main.py 4867 &
python  ${path}/main.py 4868 &
python  ${path}/main.py 4869 &
python  ${path}/main.py 4870 &
python  ${path}/main.py 4871 &
python  ${path}/main.py 4872 &
python  ${path}/main.py 4873 &
python  ${path}/main.py 4874 &
python  ${path}/main.py 4875 &
python  ${path}/main.py 4876 &
python  ${path}/main.py 4877 &
python  ${path}/main.py 4878 &
python  ${path}/main.py 4879 &
python  ${path}/main.py 4880 &
python  ${path}/main.py 4881 &
python  ${path}/main.py 4882 &
python  ${path}/main.py 4883 &
python  ${path}/main.py 4884 &
python  ${path}/main.py 4885 &
python  ${path}/main.py 4886 &
python  ${path}/main.py 4887 &
python  ${path}/main.py 4888 &
python  ${path}/main.py 4889 &
python  ${path}/main.py 4890 &
python  ${path}/main.py 4891 &
python  ${path}/main.py 4892 &
python  ${path}/main.py 4893 &
python  ${path}/main.py 4894 &
python  ${path}/main.py 4895 &
wait
