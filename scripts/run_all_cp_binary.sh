# list="5 10 15 20 25 30 35 40 45 50"
# list="60 70 80 90 100 110 120 130 140 150 175 200"
# list="155 160 165 170 180 185 190 195 225 250 275 300 350 400 500 600 750 1000" 
# list="300 305 310 315 320 325 330 335 340 345 355 360 365 370 375 380 385 390 395 400" 
list="5 10 15 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 175 200"
# list="205 210 215 220 225 230 235 240 245 250 260 270 280 290 300 310 320 330 340 350 375 400"

for cp_update_timestep in $list
do
    mkdir -p out/
    run_cmd="scripts/run_cp_binary.sh ${cp_update_timestep}"
    sbatch_cmd="sbatch ${run_cmd}"
    cmd="$sbatch_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done
