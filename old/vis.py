import numpy as np
import matplotlib.pyplot as plt
import randoms


abs_max = 1440
num_bins = 120
bin_size = abs_max // num_bins * 2
IN_FOLDER = '/scratch/groups/cslevin/eeganr/gen2annulus4/


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", type=int, default=1, help="start file num")
parser.add_argument("-e", "--end", type=int, default=60, help="end file num")
parser.add_argument("-r", "--real", action="store_true", help="uses real detector indices")
args = parser.parse_args()




bars = np.linspace(-abs_max, abs_max, num_bins, endpoint=False)

coin_fname = '/scratch/groups/cslevin/eeganr/gen2annulus3/split/11_15_coin.lm'
coin_counts = randoms.hist_tof(coin_fname, abs_max, num_bins)
del_fname = '/scratch/groups/cslevin/eeganr/gen2annulus3/split/11_15_delay.lm'
del_counts = randoms.hist_tof(del_fname, abs_max, num_bins)
act_fname = '/scratch/groups/cslevin/eeganr/gen2annulus3/split/11_15_actual.lm'
act_counts = randoms.hist_tof(act_fname, abs_max, num_bins)

# np.save('coin_hist.npy', coin_counts)
# np.save('del_hist.npy', del_counts)
# np.save('act_hist.npy', act_counts)

# coin_counts = np.load('coin_hist.npy')
# del_counts = np.load('del_hist.npy')
# act_counts = np.load('act_hist.npy')

print('making hist')
# plt.hist(tofs, bins=bars, alpha=0.4, color='#D60270')
# plt.hist(deltofs, bins=bars, alpha=0.4, color='#0038A8')

plt.bar(bars, coin_counts, alpha=0.2, color='#D60270', width=bin_size, align='edge')
plt.bar(bars, del_counts, alpha=0.2, color='#0038A8', width=bin_size, align='edge')
plt.bar(bars, act_counts, alpha=0.2, color='orange', width=bin_size, align='edge')
print('done making hist')
plt.xlabel('Time of Flight (mm)')
plt.ylabel('Counts')
plt.title('TOF Histogram')
plt.grid()
plt.savefig("my_plot.png")



# actual randoms = 1909451046.0
# DW randoms: =    1062373739


