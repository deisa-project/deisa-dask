#!/bin/bash
SCHEFILE=scheduler.json
DASK_NB_WORKERS=1                # Number of Dask workers
DASK_NB_THREAD_PER_WORKER=0      # Number of threads per Dask workers
NB_CLIENTS=(1 2 8 16)
DATA_SIZES=(16 32 64)
NB_RUNS=3
SCATTER_TYPE=("original" "optim1") # original, optim1

export DASK_DISTRIBUTED__CLIENT__HEARTBEAT=600
export DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL=600
#export DASK_DISTRIBUTED__LOGGING=DEBUG


# Launch Dask Scheduler in a 1 Node and save the connection information in $SCHEFILE
echo launching Scheduler
dask scheduler \
    --no-dashboard \
    --interface lo \
    --scheduler-file=$SCHEFILE &
dask_sch_pid=$!

# Wait for the SCHEFILE to be created
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo -n .
done

echo Scheduler booted, launching workers
dask worker \
    --interface lo \
    --no-dashboard \
    --no-nanny \
    --nworkers ${DASK_NB_WORKERS} \
    --nthreads ${DASK_NB_THREAD_PER_WORKER} \
    --local-directory /tmp \
    --scheduler-file=${SCHEFILE} &
dask_worker_pid=$!

sleep 2

for data_size in "${DATA_SIZES[@]}"; do
  for scatter_type in "${SCATTER_TYPE[@]}"; do
    for nb_clients in "${NB_CLIENTS[@]}"; do
      for run_id in $(seq 1 ${NB_RUNS}); do

        echo "======================"
        echo "START OF RUN ${run_id}/${NB_RUNS}"
        echo "scatter_type: ${scatter_type}"
        echo "data_size: ${data_size}"
        echo "nb_clients: ${nb_clients}"
        echo "======================"

        for id in $(seq 0 $(("${nb_clients}" - 1))); do
          # scheduler file, nb clients, data size, run_id, unique id
          echo -n "[${id}] running python... "
          python3 ./connect-clients.py $SCHEFILE "${nb_clients}" "${data_size}" "${run_id}" "${id}" "${scatter_type}" &
          echo "done"
          pids+=($!)
        done

        echo -n "waiting for pids... "
        wait "${pids[@]}" # wait for all background tasks
        echo "done."
        sleep 5
      done
    done
  done
done

#python3 ./connect-clients.py $SCHEFILE 10 ${DATA_SIZE}

sleep 1

kill -9 ${dask_worker_pid} ${dask_sch_pid}
rm $SCHEFILE

echo "done"