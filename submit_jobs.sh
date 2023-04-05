for i in {1..2}
do 
echo "LM_jobs_submitting" ${i}
sbatch jobs/LM_job_${i}.slurm 
sleep 25
done
