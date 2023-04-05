#!/bin/bash -l
#SBATCH -J NN_60
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time 72:00:00 # 12 hours
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rs25.iitr@gmail.com

path='/work/lcvmm/rsharma/MSK/MSK_ML_gamma/'
python  ${path}/main.py 4248 &
python  ${path}/main.py 4249 &
python  ${path}/main.py 4250 &
python  ${path}/main.py 4251 &
python  ${path}/main.py 4252 &
python  ${path}/main.py 4253 &
python  ${path}/main.py 4254 &
python  ${path}/main.py 4255 &
python  ${path}/main.py 4256 &
python  ${path}/main.py 4257 &
python  ${path}/main.py 4258 &
python  ${path}/main.py 4259 &
python  ${path}/main.py 4260 &
python  ${path}/main.py 4261 &
python  ${path}/main.py 4262 &
python  ${path}/main.py 4263 &
python  ${path}/main.py 4264 &
python  ${path}/main.py 4265 &
python  ${path}/main.py 4266 &
python  ${path}/main.py 4267 &
python  ${path}/main.py 4268 &
python  ${path}/main.py 4269 &
python  ${path}/main.py 4270 &
python  ${path}/main.py 4271 &
python  ${path}/main.py 4272 &
python  ${path}/main.py 4273 &
python  ${path}/main.py 4274 &
python  ${path}/main.py 4275 &
python  ${path}/main.py 4276 &
python  ${path}/main.py 4277 &
python  ${path}/main.py 4278 &
python  ${path}/main.py 4279 &
python  ${path}/main.py 4280 &
python  ${path}/main.py 4281 &
python  ${path}/main.py 4282 &
python  ${path}/main.py 4283 &
python  ${path}/main.py 4284 &
python  ${path}/main.py 4285 &
python  ${path}/main.py 4286 &
python  ${path}/main.py 4287 &
python  ${path}/main.py 4288 &
python  ${path}/main.py 4289 &
python  ${path}/main.py 4290 &
python  ${path}/main.py 4291 &
python  ${path}/main.py 4292 &
python  ${path}/main.py 4293 &
python  ${path}/main.py 4294 &
python  ${path}/main.py 4295 &
python  ${path}/main.py 4296 &
python  ${path}/main.py 4297 &
python  ${path}/main.py 4298 &
python  ${path}/main.py 4299 &
python  ${path}/main.py 4300 &
python  ${path}/main.py 4301 &
python  ${path}/main.py 4302 &
python  ${path}/main.py 4303 &
python  ${path}/main.py 4304 &
python  ${path}/main.py 4305 &
python  ${path}/main.py 4306 &
python  ${path}/main.py 4307 &
python  ${path}/main.py 4308 &
python  ${path}/main.py 4309 &
python  ${path}/main.py 4310 &
python  ${path}/main.py 4311 &
python  ${path}/main.py 4312 &
python  ${path}/main.py 4313 &
python  ${path}/main.py 4314 &
python  ${path}/main.py 4315 &
python  ${path}/main.py 4316 &
python  ${path}/main.py 4317 &
python  ${path}/main.py 4318 &
python  ${path}/main.py 4319 &
wait
