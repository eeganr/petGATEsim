import numpy as np
import json

MODULES = 16

casey_file = '/scratch/groups/cslevin/eeganr/cylwater/cylwat_eval/split/result_v14/validation_summary.json'

coin_file = '/scratch/groups/cslevin/eeganr/cylwater/cylwat_nocorr/coin_lor.npy'
delay_file = '/scratch/groups/cslevin/eeganr/cylwater/cylwat_nocorr/dw_nums.npy'
gt_file = '/scratch/groups/cslevin/eeganr/cylwater/cylwat_nocorr/actuals.npy'

with open(casey_file, 'r') as f:
    casey_json = json.loads(f.read())

def pool_LOR(array):
    CRYS_PER_DET = array.shape[0] // MODULES
    arr_lor = array.reshape(MODULES, CRYS_PER_DET, MODULES, CRYS_PER_DET).sum(axis=(1, 3))
    return arr_lor

coin = pool_LOR(np.load(coin_file, mmap_mode='r'))
delay = pool_LOR(np.load(delay_file, mmap_mode='r')) / coin
gt = pool_LOR(np.load(gt_file, mmap_mode='r')) / coin

casey = np.zeros(gt.shape)
tof = np.zeros(gt.shape)
for lor in casey_json['module_pair_results']:
    casey[lor['sub0']][lor['sub1']] = lor['casey_estimated_rf']
    casey[lor['sub1']][lor['sub0']] = lor['casey_estimated_rf']
    tof[lor['sub1']][lor['sub0']] = lor['tof_estimated_rf']
    tof[lor['sub1']][lor['sub0']] = lor['tof_estimated_rf']

casey_perf = casey / gt
delay_perf = delay / gt

np.save('/scratch/groups/cslevin/eeganr/cylwater/cylwat_eval/gt_lut.npy', gt)
np.save('/scratch/groups/cslevin/eeganr/cylwater/cylwat_eval/coin_lut.npy', coin)

