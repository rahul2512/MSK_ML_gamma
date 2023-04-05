#!/bin/bash -l
#SBATCH -J NN_128
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 9144 &
python  ${path}/main.py 9145 &
python  ${path}/main.py 9146 &
python  ${path}/main.py 9147 &
python  ${path}/main.py 9148 &
python  ${path}/main.py 9149 &
python  ${path}/main.py 9150 &
python  ${path}/main.py 9151 &
python  ${path}/main.py 9152 &
python  ${path}/main.py 9153 &
python  ${path}/main.py 9154 &
python  ${path}/main.py 9155 &
python  ${path}/main.py 9156 &
python  ${path}/main.py 9157 &
python  ${path}/main.py 9158 &
python  ${path}/main.py 9159 &
python  ${path}/main.py 9160 &
python  ${path}/main.py 9161 &
python  ${path}/main.py 9162 &
python  ${path}/main.py 9163 &
python  ${path}/main.py 9164 &
python  ${path}/main.py 9165 &
python  ${path}/main.py 9166 &
python  ${path}/main.py 9167 &
python  ${path}/main.py 9168 &
python  ${path}/main.py 9169 &
python  ${path}/main.py 9170 &
python  ${path}/main.py 9171 &
python  ${path}/main.py 9172 &
python  ${path}/main.py 9173 &
python  ${path}/main.py 9174 &
python  ${path}/main.py 9175 &
python  ${path}/main.py 9176 &
python  ${path}/main.py 9177 &
python  ${path}/main.py 9178 &
python  ${path}/main.py 9179 &
python  ${path}/main.py 9180 &
python  ${path}/main.py 9181 &
python  ${path}/main.py 9182 &
python  ${path}/main.py 9183 &
python  ${path}/main.py 9184 &
python  ${path}/main.py 9185 &
python  ${path}/main.py 9186 &
python  ${path}/main.py 9187 &
python  ${path}/main.py 9188 &
python  ${path}/main.py 9189 &
python  ${path}/main.py 9190 &
python  ${path}/main.py 9191 &
python  ${path}/main.py 9192 &
python  ${path}/main.py 9193 &
python  ${path}/main.py 9194 &
python  ${path}/main.py 9195 &
python  ${path}/main.py 9196 &
python  ${path}/main.py 9197 &
python  ${path}/main.py 9198 &
python  ${path}/main.py 9199 &
python  ${path}/main.py 9200 &
python  ${path}/main.py 9201 &
python  ${path}/main.py 9202 &
python  ${path}/main.py 9203 &
python  ${path}/main.py 9204 &
python  ${path}/main.py 9205 &
python  ${path}/main.py 9206 &
python  ${path}/main.py 9207 &
python  ${path}/main.py 9208 &
python  ${path}/main.py 9209 &
python  ${path}/main.py 9210 &
python  ${path}/main.py 9211 &
python  ${path}/main.py 9212 &
python  ${path}/main.py 9213 &
python  ${path}/main.py 9214 &
python  ${path}/main.py 9215 &
wait
