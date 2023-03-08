baselines="noam giam"
for baseline in $baselines
do
    mkdir -p out/
    run_cmd="scripts/run_baselines.sh ${baseline}"
    sbatch_cmd="sbatch ${run_cmd}"
    cmd="$sbatch_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done
