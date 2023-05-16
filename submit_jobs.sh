for i in {1..6}
do 
echo "xgbr_jobs_submitting" ${i}
sbatch jobs/xgbr_job_${i}.slurm 
sleep 10
done
