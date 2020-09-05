import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_results(fname_data):

    with open(fname_data, 'rb') as f:
        data = pickle.load(f)

    clusters = data[0]
    energies_all = data[1]
    energy_nlce = data[2]
    orders_energy = data[3]
    weight_energy = data[4]
    energy_nlce_euler_resum = data[5]
    entropies_all = data[6]
    entropy_nlce = data[7]
    orders_entropy = data[8]
    weight_entropy = data[9]
    entropy_nlce_euler_resum = data[10]
    specific_heats_all = data[11]
    specific_heat_nlce = data[12]
    orders_specific_heat = data[13]
    weight_specific_heat = data[14]
    spheat_nlce_euler_resum = data[15]
    szsz_corrs_all = data[16]
    szsz_nlce = data[17]
    orders_szsz = data[18]
    weight_szsz = data[19]
    szsz_nlce_euler_resum = data[20]
    temps = data[21]

    fig_handle = plt.figure()
    nrows = 2
    ncols = 2

    plt.subplot(nrows, ncols, 1)
    leg = []
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_energy[0: ii + 1, ...], 0))
        order_str = 'nlce order = %d' % (ii + 1)
        leg.append(order_str)
    leg.append('euler resum')
    plt.semilogx(temps, energy_nlce_euler_resum)

    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Energy/site (J)')
    plt.title('Energy/site')
    plt.ylim([-0.8, 0])
    plt.legend(leg)

    plt.subplot(nrows, ncols, 2)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_entropy[0: ii + 1, ...], 0))
    plt.semilogx(temps, entropy_nlce_euler_resum)
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Entropy/site ()')
    plt.ylim([0, 0.8])
    # plt.title('Entropy/site')

    plt.subplot(nrows, ncols, 3)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_specific_heat[0: ii + 1, ...], 0))
    plt.semilogx(temps, spheat_nlce_euler_resum)
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('Specific heat/site ()')
    plt.ylim([0, 0.6])
    # plt.title('Specific heat / site')

    plt.subplot(nrows, ncols, 4)
    # plt.semilogx(temps, szsz_full)
    for ii in range(4, orders_energy.shape[0]):
        plt.semilogx(temps, np.sum(orders_szsz[0: ii + 1, ...], 0))
    plt.semilogx(temps, szsz_nlce_euler_resum)
    plt.grid()
    plt.xlabel('Temperature (J)')
    plt.ylabel('SzSz mean ()')
    plt.ylim([-1, 1])

    plt.show()

    return fig_handle

if __name__ == "__main__":
    plot_results(sys.argv[1])