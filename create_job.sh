##  1270080 total number of hyp
rm jobs/LM_job*slurm
count=0
for (( i=0; i<55; i+=1 )) 
do
index=$[count/36+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> LM_job_${index}.slurm
done 

for index in {1..2}
do 
touch tmp 
cat heading.txt >> tmp
cat LM_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm LM_job_${index}.slurm
mv tmp jobs/LM_job_${index}.slurm
sed -i "s/MMM/LM_${index}/g" jobs/LM_job_${index}.slurm
done
