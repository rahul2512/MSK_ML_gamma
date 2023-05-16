#!/bin/bash -l
#SBATCH -J xgbr_4
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 216 &
python  ${path}/main.py 217 &
python  ${path}/main.py 218 &
python  ${path}/main.py 219 &
python  ${path}/main.py 220 &
python  ${path}/main.py 221 &
python  ${path}/main.py 222 &
python  ${path}/main.py 223 &
python  ${path}/main.py 224 &
python  ${path}/main.py 225 &
python  ${path}/main.py 226 &
python  ${path}/main.py 227 &
python  ${path}/main.py 228 &
python  ${path}/main.py 229 &
python  ${path}/main.py 230 &
python  ${path}/main.py 231 &
python  ${path}/main.py 232 &
python  ${path}/main.py 233 &
python  ${path}/main.py 234 &
python  ${path}/main.py 235 &
python  ${path}/main.py 236 &
python  ${path}/main.py 237 &
python  ${path}/main.py 238 &
python  ${path}/main.py 239 &
python  ${path}/main.py 240 &
python  ${path}/main.py 241 &
python  ${path}/main.py 242 &
python  ${path}/main.py 243 &
python  ${path}/main.py 244 &
python  ${path}/main.py 245 &
python  ${path}/main.py 246 &
python  ${path}/main.py 247 &
python  ${path}/main.py 248 &
python  ${path}/main.py 249 &
python  ${path}/main.py 250 &
python  ${path}/main.py 251 &
python  ${path}/main.py 252 &
python  ${path}/main.py 253 &
python  ${path}/main.py 254 &
python  ${path}/main.py 255 &
python  ${path}/main.py 256 &
python  ${path}/main.py 257 &
python  ${path}/main.py 258 &
python  ${path}/main.py 259 &
python  ${path}/main.py 260 &
python  ${path}/main.py 261 &
python  ${path}/main.py 262 &
python  ${path}/main.py 263 &
python  ${path}/main.py 264 &
python  ${path}/main.py 265 &
python  ${path}/main.py 266 &
python  ${path}/main.py 267 &
python  ${path}/main.py 268 &
python  ${path}/main.py 269 &
python  ${path}/main.py 270 &
python  ${path}/main.py 271 &
python  ${path}/main.py 272 &
python  ${path}/main.py 273 &
python  ${path}/main.py 274 &
python  ${path}/main.py 275 &
python  ${path}/main.py 276 &
python  ${path}/main.py 277 &
python  ${path}/main.py 278 &
python  ${path}/main.py 279 &
python  ${path}/main.py 280 &
python  ${path}/main.py 281 &
python  ${path}/main.py 282 &
python  ${path}/main.py 283 &
python  ${path}/main.py 284 &
python  ${path}/main.py 285 &
python  ${path}/main.py 286 &
python  ${path}/main.py 287 &
wait
