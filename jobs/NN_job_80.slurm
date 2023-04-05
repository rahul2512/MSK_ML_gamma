#!/bin/bash -l
#SBATCH -J NN_80
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 5688 &
python  ${path}/main.py 5689 &
python  ${path}/main.py 5690 &
python  ${path}/main.py 5691 &
python  ${path}/main.py 5692 &
python  ${path}/main.py 5693 &
python  ${path}/main.py 5694 &
python  ${path}/main.py 5695 &
python  ${path}/main.py 5696 &
python  ${path}/main.py 5697 &
python  ${path}/main.py 5698 &
python  ${path}/main.py 5699 &
python  ${path}/main.py 5700 &
python  ${path}/main.py 5701 &
python  ${path}/main.py 5702 &
python  ${path}/main.py 5703 &
python  ${path}/main.py 5704 &
python  ${path}/main.py 5705 &
python  ${path}/main.py 5706 &
python  ${path}/main.py 5707 &
python  ${path}/main.py 5708 &
python  ${path}/main.py 5709 &
python  ${path}/main.py 5710 &
python  ${path}/main.py 5711 &
python  ${path}/main.py 5712 &
python  ${path}/main.py 5713 &
python  ${path}/main.py 5714 &
python  ${path}/main.py 5715 &
python  ${path}/main.py 5716 &
python  ${path}/main.py 5717 &
python  ${path}/main.py 5718 &
python  ${path}/main.py 5719 &
python  ${path}/main.py 5720 &
python  ${path}/main.py 5721 &
python  ${path}/main.py 5722 &
python  ${path}/main.py 5723 &
python  ${path}/main.py 5724 &
python  ${path}/main.py 5725 &
python  ${path}/main.py 5726 &
python  ${path}/main.py 5727 &
python  ${path}/main.py 5728 &
python  ${path}/main.py 5729 &
python  ${path}/main.py 5730 &
python  ${path}/main.py 5731 &
python  ${path}/main.py 5732 &
python  ${path}/main.py 5733 &
python  ${path}/main.py 5734 &
python  ${path}/main.py 5735 &
python  ${path}/main.py 5736 &
python  ${path}/main.py 5737 &
python  ${path}/main.py 5738 &
python  ${path}/main.py 5739 &
python  ${path}/main.py 5740 &
python  ${path}/main.py 5741 &
python  ${path}/main.py 5742 &
python  ${path}/main.py 5743 &
python  ${path}/main.py 5744 &
python  ${path}/main.py 5745 &
python  ${path}/main.py 5746 &
python  ${path}/main.py 5747 &
python  ${path}/main.py 5748 &
python  ${path}/main.py 5749 &
python  ${path}/main.py 5750 &
python  ${path}/main.py 5751 &
python  ${path}/main.py 5752 &
python  ${path}/main.py 5753 &
python  ${path}/main.py 5754 &
python  ${path}/main.py 5755 &
python  ${path}/main.py 5756 &
python  ${path}/main.py 5757 &
python  ${path}/main.py 5758 &
python  ${path}/main.py 5759 &
wait
