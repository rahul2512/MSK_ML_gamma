for i in {5..25}
do 
echo "CNNLSTN_jobs_submitting" ${i}
sbatch jobs/CNNLSTM_job_${i}.slurm 
sleep 5
done
