model="xgbr"
ncpu=72
##  1270080 total number of hyp
filename="./hyperparameters/hyperparam_""${model}"".txt"
linecount=$(wc -l < "$filename")
echo ${linecount} "total number of jobs"
rm jobs/${model}_job*slurm
count=0
for (( i=0; i<${linecount}; i+=1 )) 
do
index=$[count/ncpu+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> ${model}_job_${index}.slurm
done 


nfile=$((linecount / ncpu))
echo ${nfile} "total number of jobs to submit"

for index in $(seq 1 "$nfile")
do 
touch tmp 
cat heading.txt >> tmp
cat ${model}_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm ${model}_job_${index}.slurm
mv tmp jobs/${model}_job_${index}.slurm
sed -i "s/MMM/${model}_${index}/g" jobs/${model}_job_${index}.slurm
done

cp template_submit_jobs.sh submit_jobs.sh 
sed -i "s/XXX/${nfile}/g" submit_jobs.sh
sed -i "s/YYY/${model}/g" submit_jobs.sh
sed -i "s/UUUU/${model}/g" main.py
