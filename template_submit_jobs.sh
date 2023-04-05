for i in {1..XXX}
do 
echo "YYY_jobs_submitting" ${i}
sbatch jobs/YYY_job_${i}.slurm 
sleep 25
done
