# list="5 10 15 20 25 30 35 40 45 50"
list="60 70 80 90 100 110 120 130 140 150 175 200"
for cp_update_timestep in $list
do
    mkdir -p out/
    run_cmd="run_cp.sh ${cp_update_timestep}"
    sbatch_cmd="sbatch ${run_cmd}"
    cmd="$sbatch_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done