#!/bin/bash -l
#SBATCH -J NN_100
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 7128 &
python  ${path}/main.py 7129 &
python  ${path}/main.py 7130 &
python  ${path}/main.py 7131 &
python  ${path}/main.py 7132 &
python  ${path}/main.py 7133 &
python  ${path}/main.py 7134 &
python  ${path}/main.py 7135 &
python  ${path}/main.py 7136 &
python  ${path}/main.py 7137 &
python  ${path}/main.py 7138 &
python  ${path}/main.py 7139 &
python  ${path}/main.py 7140 &
python  ${path}/main.py 7141 &
python  ${path}/main.py 7142 &
python  ${path}/main.py 7143 &
python  ${path}/main.py 7144 &
python  ${path}/main.py 7145 &
python  ${path}/main.py 7146 &
python  ${path}/main.py 7147 &
python  ${path}/main.py 7148 &
python  ${path}/main.py 7149 &
python  ${path}/main.py 7150 &
python  ${path}/main.py 7151 &
python  ${path}/main.py 7152 &
python  ${path}/main.py 7153 &
python  ${path}/main.py 7154 &
python  ${path}/main.py 7155 &
python  ${path}/main.py 7156 &
python  ${path}/main.py 7157 &
python  ${path}/main.py 7158 &
python  ${path}/main.py 7159 &
python  ${path}/main.py 7160 &
python  ${path}/main.py 7161 &
python  ${path}/main.py 7162 &
python  ${path}/main.py 7163 &
python  ${path}/main.py 7164 &
python  ${path}/main.py 7165 &
python  ${path}/main.py 7166 &
python  ${path}/main.py 7167 &
python  ${path}/main.py 7168 &
python  ${path}/main.py 7169 &
python  ${path}/main.py 7170 &
python  ${path}/main.py 7171 &
python  ${path}/main.py 7172 &
python  ${path}/main.py 7173 &
python  ${path}/main.py 7174 &
python  ${path}/main.py 7175 &
python  ${path}/main.py 7176 &
python  ${path}/main.py 7177 &
python  ${path}/main.py 7178 &
python  ${path}/main.py 7179 &
python  ${path}/main.py 7180 &
python  ${path}/main.py 7181 &
python  ${path}/main.py 7182 &
python  ${path}/main.py 7183 &
python  ${path}/main.py 7184 &
python  ${path}/main.py 7185 &
python  ${path}/main.py 7186 &
python  ${path}/main.py 7187 &
python  ${path}/main.py 7188 &
python  ${path}/main.py 7189 &
python  ${path}/main.py 7190 &
python  ${path}/main.py 7191 &
python  ${path}/main.py 7192 &
python  ${path}/main.py 7193 &
python  ${path}/main.py 7194 &
python  ${path}/main.py 7195 &
python  ${path}/main.py 7196 &
python  ${path}/main.py 7197 &
python  ${path}/main.py 7198 &
python  ${path}/main.py 7199 &
wait
