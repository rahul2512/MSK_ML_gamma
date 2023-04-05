#!/bin/bash -l
#SBATCH -J NN_66
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4680 &
python  ${path}/main.py 4681 &
python  ${path}/main.py 4682 &
python  ${path}/main.py 4683 &
python  ${path}/main.py 4684 &
python  ${path}/main.py 4685 &
python  ${path}/main.py 4686 &
python  ${path}/main.py 4687 &
python  ${path}/main.py 4688 &
python  ${path}/main.py 4689 &
python  ${path}/main.py 4690 &
python  ${path}/main.py 4691 &
python  ${path}/main.py 4692 &
python  ${path}/main.py 4693 &
python  ${path}/main.py 4694 &
python  ${path}/main.py 4695 &
python  ${path}/main.py 4696 &
python  ${path}/main.py 4697 &
python  ${path}/main.py 4698 &
python  ${path}/main.py 4699 &
python  ${path}/main.py 4700 &
python  ${path}/main.py 4701 &
python  ${path}/main.py 4702 &
python  ${path}/main.py 4703 &
python  ${path}/main.py 4704 &
python  ${path}/main.py 4705 &
python  ${path}/main.py 4706 &
python  ${path}/main.py 4707 &
python  ${path}/main.py 4708 &
python  ${path}/main.py 4709 &
python  ${path}/main.py 4710 &
python  ${path}/main.py 4711 &
python  ${path}/main.py 4712 &
python  ${path}/main.py 4713 &
python  ${path}/main.py 4714 &
python  ${path}/main.py 4715 &
python  ${path}/main.py 4716 &
python  ${path}/main.py 4717 &
python  ${path}/main.py 4718 &
python  ${path}/main.py 4719 &
python  ${path}/main.py 4720 &
python  ${path}/main.py 4721 &
python  ${path}/main.py 4722 &
python  ${path}/main.py 4723 &
python  ${path}/main.py 4724 &
python  ${path}/main.py 4725 &
python  ${path}/main.py 4726 &
python  ${path}/main.py 4727 &
python  ${path}/main.py 4728 &
python  ${path}/main.py 4729 &
python  ${path}/main.py 4730 &
python  ${path}/main.py 4731 &
python  ${path}/main.py 4732 &
python  ${path}/main.py 4733 &
python  ${path}/main.py 4734 &
python  ${path}/main.py 4735 &
python  ${path}/main.py 4736 &
python  ${path}/main.py 4737 &
python  ${path}/main.py 4738 &
python  ${path}/main.py 4739 &
python  ${path}/main.py 4740 &
python  ${path}/main.py 4741 &
python  ${path}/main.py 4742 &
python  ${path}/main.py 4743 &
python  ${path}/main.py 4744 &
python  ${path}/main.py 4745 &
python  ${path}/main.py 4746 &
python  ${path}/main.py 4747 &
python  ${path}/main.py 4748 &
python  ${path}/main.py 4749 &
python  ${path}/main.py 4750 &
python  ${path}/main.py 4751 &
wait
