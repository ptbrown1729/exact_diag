import sys
import pickle
import glob
import re
import numpy as np
import scipy.sparse as sp
import ed_nlce

def aggregate_subclusters(file_pattern, fname_out, fname_in):

    with open(fname_in, 'rb') as f:
        cluster_data = pickle.load(f)
    nclusters = len(cluster_data[1])
    print("read cluster data from %s" % fname_in)

    # load diagonalized cluster files_out
    files = glob.glob(file_pattern)
    if len(files) != nclusters:
        raise Exception('number of cluster data files_out is not equal to the number of clusters. %d data files_out and %d clusters.', len(files), nclusters)
    print("found %d files_out matching %s" % (len(files), file_pattern))

    # sort these by number ... currently this is fragile because I'm assuming that there are no other
    # numbers in the file name besides for the cluster number.
    expr = '[^\d]*(\d+)[^\d]*'
    nums = np.zeros(len(files))
    for ii, file in enumerate(files):
        match = re.match(expr, file)
        nums[ii] = int(match.group(1))
    indices_sorted = np.argsort(nums)

    files = [file for _, file in sorted(zip(nums, files), key=lambda tuple: tuple[0])]
    nums = nums[indices_sorted]

    cluster_mult_mat = sp.csr_matrix((nclusters, nclusters))

    for file in files:
        with open(file, 'rb') as f:
            file_data = pickle.load(f)
        basis_change_mat = file_data[1]

        row_sums = np.asarray(np.sum(cluster_mult_mat, 1))
        row_sums = 1 - (row_sums > 0)
        a = row_sums.reshape([len(row_sums), ]).tolist()
        b = sp.diags(a, offsets=0, format='csr')
        cluster_mult_mat = cluster_mult_mat + b.dot(basis_change_mat)

    # TODO: does it make more sense to save this back to the first file???
    cluster_data.append(cluster_mult_mat)
    with open(fname_out, 'wb') as f:
        pickle.dump(cluster_data, f)
    print("saved subcluster information to %s" % fname_out)


if __name__ == "__main__":
    file_pattern = sys.argv[1]
    fname_out = sys.argv[2]
    fname_in = sys.argv[3]
    aggregate_subclusters(file_pattern, fname_out, fname_in)