#!/bin/bash -l
#SBATCH -J CNNLSTM_17
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1152 &
python  ${path}/main.py 1153 &
python  ${path}/main.py 1154 &
python  ${path}/main.py 1155 &
python  ${path}/main.py 1156 &
python  ${path}/main.py 1157 &
python  ${path}/main.py 1158 &
python  ${path}/main.py 1159 &
python  ${path}/main.py 1160 &
python  ${path}/main.py 1161 &
python  ${path}/main.py 1162 &
python  ${path}/main.py 1163 &
python  ${path}/main.py 1164 &
python  ${path}/main.py 1165 &
python  ${path}/main.py 1166 &
python  ${path}/main.py 1167 &
python  ${path}/main.py 1168 &
python  ${path}/main.py 1169 &
python  ${path}/main.py 1170 &
python  ${path}/main.py 1171 &
python  ${path}/main.py 1172 &
python  ${path}/main.py 1173 &
python  ${path}/main.py 1174 &
python  ${path}/main.py 1175 &
python  ${path}/main.py 1176 &
python  ${path}/main.py 1177 &
python  ${path}/main.py 1178 &
python  ${path}/main.py 1179 &
python  ${path}/main.py 1180 &
python  ${path}/main.py 1181 &
python  ${path}/main.py 1182 &
python  ${path}/main.py 1183 &
python  ${path}/main.py 1184 &
python  ${path}/main.py 1185 &
python  ${path}/main.py 1186 &
python  ${path}/main.py 1187 &
python  ${path}/main.py 1188 &
python  ${path}/main.py 1189 &
python  ${path}/main.py 1190 &
python  ${path}/main.py 1191 &
python  ${path}/main.py 1192 &
python  ${path}/main.py 1193 &
python  ${path}/main.py 1194 &
python  ${path}/main.py 1195 &
python  ${path}/main.py 1196 &
python  ${path}/main.py 1197 &
python  ${path}/main.py 1198 &
python  ${path}/main.py 1199 &
python  ${path}/main.py 1200 &
python  ${path}/main.py 1201 &
python  ${path}/main.py 1202 &
python  ${path}/main.py 1203 &
python  ${path}/main.py 1204 &
python  ${path}/main.py 1205 &
python  ${path}/main.py 1206 &
python  ${path}/main.py 1207 &
python  ${path}/main.py 1208 &
python  ${path}/main.py 1209 &
python  ${path}/main.py 1210 &
python  ${path}/main.py 1211 &
python  ${path}/main.py 1212 &
python  ${path}/main.py 1213 &
python  ${path}/main.py 1214 &
python  ${path}/main.py 1215 &
python  ${path}/main.py 1216 &
python  ${path}/main.py 1217 &
python  ${path}/main.py 1218 &
python  ${path}/main.py 1219 &
python  ${path}/main.py 1220 &
python  ${path}/main.py 1221 &
python  ${path}/main.py 1222 &
python  ${path}/main.py 1223 &
wait
