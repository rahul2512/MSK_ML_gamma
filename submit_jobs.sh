for i in {1..25}
do 
echo "CNNLSTM_jobs_submitting" ${i}
sbatch jobs/CNNLSTM_job_${i}.slurm 
sleep 5
done
