#!/bin/bash -l
#SBATCH -J CNNLSTM_18
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 1224 &
python  ${path}/main.py 1225 &
python  ${path}/main.py 1226 &
python  ${path}/main.py 1227 &
python  ${path}/main.py 1228 &
python  ${path}/main.py 1229 &
python  ${path}/main.py 1230 &
python  ${path}/main.py 1231 &
python  ${path}/main.py 1232 &
python  ${path}/main.py 1233 &
python  ${path}/main.py 1234 &
python  ${path}/main.py 1235 &
python  ${path}/main.py 1236 &
python  ${path}/main.py 1237 &
python  ${path}/main.py 1238 &
python  ${path}/main.py 1239 &
python  ${path}/main.py 1240 &
python  ${path}/main.py 1241 &
python  ${path}/main.py 1242 &
python  ${path}/main.py 1243 &
python  ${path}/main.py 1244 &
python  ${path}/main.py 1245 &
python  ${path}/main.py 1246 &
python  ${path}/main.py 1247 &
python  ${path}/main.py 1248 &
python  ${path}/main.py 1249 &
python  ${path}/main.py 1250 &
python  ${path}/main.py 1251 &
python  ${path}/main.py 1252 &
python  ${path}/main.py 1253 &
python  ${path}/main.py 1254 &
python  ${path}/main.py 1255 &
python  ${path}/main.py 1256 &
python  ${path}/main.py 1257 &
python  ${path}/main.py 1258 &
python  ${path}/main.py 1259 &
python  ${path}/main.py 1260 &
python  ${path}/main.py 1261 &
python  ${path}/main.py 1262 &
python  ${path}/main.py 1263 &
python  ${path}/main.py 1264 &
python  ${path}/main.py 1265 &
python  ${path}/main.py 1266 &
python  ${path}/main.py 1267 &
python  ${path}/main.py 1268 &
python  ${path}/main.py 1269 &
python  ${path}/main.py 1270 &
python  ${path}/main.py 1271 &
python  ${path}/main.py 1272 &
python  ${path}/main.py 1273 &
python  ${path}/main.py 1274 &
python  ${path}/main.py 1275 &
python  ${path}/main.py 1276 &
python  ${path}/main.py 1277 &
python  ${path}/main.py 1278 &
python  ${path}/main.py 1279 &
python  ${path}/main.py 1280 &
python  ${path}/main.py 1281 &
python  ${path}/main.py 1282 &
python  ${path}/main.py 1283 &
python  ${path}/main.py 1284 &
python  ${path}/main.py 1285 &
python  ${path}/main.py 1286 &
python  ${path}/main.py 1287 &
python  ${path}/main.py 1288 &
python  ${path}/main.py 1289 &
python  ${path}/main.py 1290 &
python  ${path}/main.py 1291 &
python  ${path}/main.py 1292 &
python  ${path}/main.py 1293 &
python  ${path}/main.py 1294 &
python  ${path}/main.py 1295 &
wait
