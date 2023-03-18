##  1270080 total number of hyp
rm jobs/RNN_job*slurm
count=0
for (( i=0; i<5185; i+=1 )) 
do
index=$[count/36+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> RNN_job_${index}.slurm
done 

for index in {1..145}
do 
touch tmp 
cat heading.txt >> tmp
cat RNN_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm RNN_job_${index}.slurm
mv tmp jobs/RNN_job_${index}.slurm
sed -i "s/MMM/RNN_${index}/g" jobs/RNN_job_${index}.slurm
done
