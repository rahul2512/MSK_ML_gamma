#!/bin/bash -l
#SBATCH -J NN_156
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 11160 &
python  ${path}/main.py 11161 &
python  ${path}/main.py 11162 &
python  ${path}/main.py 11163 &
python  ${path}/main.py 11164 &
python  ${path}/main.py 11165 &
python  ${path}/main.py 11166 &
python  ${path}/main.py 11167 &
python  ${path}/main.py 11168 &
python  ${path}/main.py 11169 &
python  ${path}/main.py 11170 &
python  ${path}/main.py 11171 &
python  ${path}/main.py 11172 &
python  ${path}/main.py 11173 &
python  ${path}/main.py 11174 &
python  ${path}/main.py 11175 &
python  ${path}/main.py 11176 &
python  ${path}/main.py 11177 &
python  ${path}/main.py 11178 &
python  ${path}/main.py 11179 &
python  ${path}/main.py 11180 &
python  ${path}/main.py 11181 &
python  ${path}/main.py 11182 &
python  ${path}/main.py 11183 &
python  ${path}/main.py 11184 &
python  ${path}/main.py 11185 &
python  ${path}/main.py 11186 &
python  ${path}/main.py 11187 &
python  ${path}/main.py 11188 &
python  ${path}/main.py 11189 &
python  ${path}/main.py 11190 &
python  ${path}/main.py 11191 &
python  ${path}/main.py 11192 &
python  ${path}/main.py 11193 &
python  ${path}/main.py 11194 &
python  ${path}/main.py 11195 &
python  ${path}/main.py 11196 &
python  ${path}/main.py 11197 &
python  ${path}/main.py 11198 &
python  ${path}/main.py 11199 &
python  ${path}/main.py 11200 &
python  ${path}/main.py 11201 &
python  ${path}/main.py 11202 &
python  ${path}/main.py 11203 &
python  ${path}/main.py 11204 &
python  ${path}/main.py 11205 &
python  ${path}/main.py 11206 &
python  ${path}/main.py 11207 &
python  ${path}/main.py 11208 &
python  ${path}/main.py 11209 &
python  ${path}/main.py 11210 &
python  ${path}/main.py 11211 &
python  ${path}/main.py 11212 &
python  ${path}/main.py 11213 &
python  ${path}/main.py 11214 &
python  ${path}/main.py 11215 &
python  ${path}/main.py 11216 &
python  ${path}/main.py 11217 &
python  ${path}/main.py 11218 &
python  ${path}/main.py 11219 &
python  ${path}/main.py 11220 &
python  ${path}/main.py 11221 &
python  ${path}/main.py 11222 &
python  ${path}/main.py 11223 &
python  ${path}/main.py 11224 &
python  ${path}/main.py 11225 &
python  ${path}/main.py 11226 &
python  ${path}/main.py 11227 &
python  ${path}/main.py 11228 &
python  ${path}/main.py 11229 &
python  ${path}/main.py 11230 &
python  ${path}/main.py 11231 &
wait
