#import cPickle as pickle
import pickle
import os
import sys
from ed_clusters import *

def generate_clusters(fname_cluster_dat, max_cluster_order):

    # if data file already exists, read number of clusters and return
    if os.path.isfile(fname_cluster_dat):
        with open(fname_cluster_dat, 'rb') as f:
            data = pickle.load(f)
        cluster_list = data[2]
        return len(cluster_list)

    # if data file doesn't already exist, we must generate the clusters and create the file
    clusters_list, cluster_multiplicities, sub_cluster_mult, order_start_indices = \
            get_all_clusters_with_subclusters(max_cluster_order)

    cluster_multiplicities = cluster_multiplicities[None, :]

    # save cluster data
    data_clusters = [max_cluster_order, cluster_multiplicities, clusters_list, sub_cluster_mult, order_start_indices]
    with open(fname_cluster_dat, 'wb') as f:
        pickle.dump(data_clusters, f)

    return len(clusters_list)

if __name__ == "__main__":
    output_fname = sys.argv[1]
    max_cluster_order = int(sys.argv[2])
    num_clusters = generate_clusters(output_fname, max_cluster_order)
    print num_clusters