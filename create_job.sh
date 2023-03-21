##  1270080 total number of hyp
rm jobs/CNNLSTM_job*slurm
count=0
for (( i=0; i<1729; i+=1 )) 
do
index=$[count/36+1]
count=$[count+1]
echo "python " '${path}'"/main.py" ${i} "&" >> CNNLSTM_job_${index}.slurm
done 

for index in {1..49}
do 
touch tmp 
cat heading.txt >> tmp
cat CNNLSTM_job_${index}.slurm >> tmp 
echo "wait" >> tmp
rm CNNLSTM_job_${index}.slurm
mv tmp jobs/CNNLSTM_job_${index}.slurm
sed -i "s/MMM/CNNLSTM_${index}/g" jobs/CNNLSTM_job_${index}.slurm
done
