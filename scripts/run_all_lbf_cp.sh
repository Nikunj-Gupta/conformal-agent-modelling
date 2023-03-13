list="250 500 750 1000 1250 1500 1750 2000"

for cp_update_timestep in $list
do
    mkdir -p out/
    run_cmd="scripts/run_lbf_cp.sh ${cp_update_timestep}"
    sbatch_cmd="sbatch ${run_cmd}"
    cmd="$sbatch_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done
