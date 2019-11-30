#!/bin/sh

# settings and file paths
nlce_order=9
root_path="/scratch/ptbrown/exact_diag"

# generate clusters data
cluster_dat_script="$root_path/generate_clusters_no_subs.py"
cluster_dat_log="$root_path/generate_cluster_dat.log"
fname_cluster_dat="$root_path/cluster_data_order=$nlce_order.dat"
slurm_get_clusters="$root_path/slurm_single.sh"

# generate sub clusters data
sub_cluster_dat_script="$root_path/find_subclusters.py"
slurm_subclusters="$root_path/slurm_array.sh"
data_prefix="subcluster"
subcluster_aggregate_data="$root_path/subcluster_data_order=$nlce_order.dat"
subcluster_aggregate_script="$root_path/aggregate_subclusters.py"
subcluster_dat_log="$root_path/generate_subcluster_dat.log"

# diagonalize clusters data
diag_cluster_script="$root_path/diag_cluster.py"
slurm_diag="$root_path/slurm_array.sh"
diag_file_prefix="diag_cluster"
aggregate_diag_script="$root_path/aggregate_results.py"
nlce_results_fname="$root_path/nlce_results_order=$nlce_order.dat"
nlce_log="$root_path/nlce_results_order=$nlce_order.log"
num_in_parallel="100"
num_limit_per_job="500"

# activate virtual environment
source /home/ptbrown/python_env/bin/activate
# generate cluster data
# uncomment the next line to test this script without sbatch
#num_clusters=$(python "$cluster_dat_script" "$fname_cluster_dat" "$nlce_order" | tee "$cluster_dat_log")

# first generate all clusters. This is not parallelizable. So should I even use sbatch???
if [ ! -f "$fname_cluster_dat" ]; then
    sbatch --export=script_arg1="$fname_cluster_dat",script_arg2="$nlce_order",script_arg3="",script_fname="$cluster_dat_script",log_fname="$cluster_dat_log" "$slurm_get_clusters"
fi

while [ ! -f "$fname_cluster_dat" ]; do
    echo "file $fname_cluster_dat does not exist. Waiting for sbatch process to create it."
    sleep 5
done

# have to run this again to get the number of clusters ... maybe there is a better way?
num_clusters=$(python "$cluster_dat_script" "$fname_cluster_dat" "$nlce_order" | tee "$cluster_dat_log")
echo "Found file $fname_cluster_dat. Total number of clusters is $num_clusters. Proceeding to determination of subclusters."

# then generate all sub-clusters. This mapping process is fully parallelizable. In fact, it can go on at the same time
# the clusters are being diagonalized
num_loops=$(($num_clusters/$num_limit_per_job+1))
echo "number of loops is: $num_loops"
for i in $( seq 1 $num_loops ); do
    offset=$(( ($i-1) * $num_limit_per_job ))
    if (( $i == $num_loops )); then
        num_clusters_todo=$(($num_clusters-$offset))
    else
        num_clusters_todo=$num_limit_per_job
    fi
    echo "runner_offset is: $offset"
    echo "clusters to run with this slurm array: $num_clusters_todo"
    sbatch --export=offset="$offset",script_arg3="$fname_cluster_dat",script_fname="$sub_cluster_dat_script",log_prefix="$data_prefix",dat_prefix="$data_prefix" --array=1-$num_clusters_todo%"$num_in_parallel" "$slurm_subclusters"
done

# finally, wait for jobs to be done, then aggregate results
while (( num_clusters != $(ls "$data_prefix"*.dat | wc -l) )); do
    echo "Waiting for all subclusters to be found."
    sleep 5
done

# aggregate the subcluster data
python "$subcluster_aggregate_script" "$data_prefix*.dat" "$subcluster_aggregate_data" "$fname_cluster_dat" > "$subcluster_dat_log"

# diagonalize all clusters. This is fully parallelizable.
# Use slurm arrays instead of submitting individual batch jobs. Slurm arrays are much faster, and avoid sending an email
# for every single cluster. When I did that, OIT locked me out of my account for sending over 2000 emails!
num_loops=$(($num_clusters/$num_limit_per_job+1))
echo "number of loops is: $num_loops"
for i in $( seq 1 $num_loops ); do
    offset=$(( ($i-1) * $num_limit_per_job ))
    if (( $i == $num_loops )); then
        num_clusters_todo=$(($num_clusters-$offset))
    else
        num_clusters_todo=$num_limit_per_job
    fi
    echo "runner_offset is: $offset"
    echo "clusters to run with this slurm array: $num_clusters_todo"
    sbatch --export=offset="$offset",script_arg3="$fname_cluster_dat",script_fname="$diag_cluster_script",log_prefix="$diag_file_prefix",dat_prefix="$diag_file_prefix" --array=1-$num_clusters_todo%"$num_in_parallel" "$slurm_diag"
done

# wait until all clusters are diagonalized, then aggregate data
while (( num_clusters != $(ls "$diag_file_prefix"*.dat | wc -l) )); do
    echo "Waiting for all clusters to be diagonalized."
    sleep 5
done

# aggregate results and do nlce
python "$aggregate_diag_script" "$diag_file_prefix*.dat" "$subcluster_aggregate_data" "$nlce_results_fname" > "$nlce_log"