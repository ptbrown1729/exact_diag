import cPickle as pickle
import os
import sys
import time
import ed_nlce

def generate_clusters_no_subs(fname_cluster_dat, max_cluster_order):

    # if data file already exists, read number of clusters and return
    if os.path.isfile(fname_cluster_dat):
        with open(fname_cluster_dat, 'rb') as f:
            data = pickle.load(f)
        cluster_list = data[1]
        return len(cluster_list)

    t_start = time.clock()
    clusters_list, cluster_multiplicities, order_start_indices = ed_nlce.get_all_clusters(max_cluster_order)
    t_end = time.clock()

    # save cluster data
    data_clusters = [max_cluster_order, clusters_list, order_start_indices, t_end - t_start]
    with open(fname_cluster_dat, 'wb') as f:
        pickle.dump(data_clusters, f)

    return len(clusters_list)

if __name__ == "__main__":
    output_fname = sys.argv[1]
    max_cluster_order = int(sys.argv[2])
    num_clusters = generate_clusters_no_subs(output_fname, max_cluster_order)
    print num_clusters