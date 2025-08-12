import os
import shutil
import numpy as np
from array import array
import numpy as np
import itertools
from lmfit.models import ExponentialModel, GaussianModel
import peakutils
import matplotlib.pyplot as plt
import math
import struct
from array import array
import numpy as np
import os
import pickle
import sys
import argparse
import shutil

pixel_num = 864

#### get params
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument(dest='dir_origin', help='Path of a folder')   
parser.add_argument('--r', dest='reverse', action='store_true', default=False,
                    help='Process data in reverse direction(default: False)')        
args = parser.parse_args()
print(args)


#### params settings
Listmode = True
RandomEstimation = True
Skew = True
TOF = True
ListmodeCombine = True
Plot = True
textOutputFmt = ['%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f']
Det_convert = np.arange(0,16, dtype = int)
speedOfLight = 299792458000  # speed of light in mm
speedOfLight_length_ps = speedOfLight * math.pow(10, -12)

#### split_data files
WDIR=args.dir_origin
Files = {}
for root, dirs, files in os.walk(WDIR):
    for name in files: 
        if  '.dat' in name and 'coin' in name:
            if name not in Files.keys():
                Files[name] = []
            Files[name] += [os.path.join(root, name)]
keys = list(Files.keys())
keys.sort(key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
print(Files)
print(keys)

#### result paths
folder = WDIR + '/result'
if not os.path.isdir(folder):
    os.mkdir(folder)
result_path = folder + '/Skew'
listmode_path = folder + '/Listmode_Rcorrect' + str(RandomEstimation*1) + '_' + 'TOF' + str(TOF*1)


#### useful functions
def stack_padding(l):
    return np.column_stack((itertools.zip_longest(*l, fillvalue=np.nan)))

if not os.path.isdir(result_path):
    os.mkdir(result_path)

if not os.path.isdir(listmode_path):
    os.mkdir(listmode_path)

#### load geometry file
if Listmode:
    with open('geometry.pickle', 'rb') as file:
        crystalPositionMap = pickle.load(file)    

        
data_all = np.zeros((864*16, 864*16), dtype = np.int16)

#### iterate all detid pairs
for f in keys:
    print(f)
    sub0 = int(f.split('_')[0])
    sub1 = int(f.split('_')[1])
    skewoffset = np.zeros((pixel_num,pixel_num), dtype = np.int16)
    
    if os.path.isfile(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected_tof_listmode.jpg'):
        continue
    data = 0
    for file in Files[f]:
        print(file)
        fo=open(file,"rb")
        data_tmp = np.fromfile(fo,dtype = np.int16)
        data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
        if isinstance(data, int):
            data = data_tmp
        else:
            data = np.concatenate([data,data_tmp], axis = 1)
        print(data.shape)
        fo.close()
    #### convert TDC bin to ps. 1 TDC bin = 1600/1024 ps
    data[2, :] = data[2, :]*1.5625
    
    #### plot the time spectrum of that detid pair
    if Plot:
        tmp = data[2,:]
        plt.hist(tmp, np.linspace(-5000,5000,201))
        lim_max = np.max(np.histogram(tmp, np.linspace(-5000,5000,201))[0]) * 1.2
        plt.ylim([0,lim_max])
        plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin.jpg')
        plt.clf()
        
    #### substract random events
    if RandomEstimation:
        f_random_corrected = result_path + '/' + str(sub0) + '_' + str(sub1) + '_random_corrected.dat'
        if not os.path.isfile(f_random_corrected):
            delay = 0
            for file in Files[f]:
                print(file.replace('coin','delay'))
                fo=open(file.replace('coin','delay'),"rb")
                data_tmp = np.fromfile(fo,dtype = np.int16)
                data_tmp = np.reshape(data_tmp, (int(data_tmp.shape[0]/3), 3)).transpose()
                if isinstance(delay, int):
                    delay = data_tmp
                else:
                    delay = np.concatenate([delay,data_tmp], axis = 1)
                print(delay.shape)
                fo.close()
            #### shift delay events to the promty window position
            delay[2, :] = delay[2, :]*1.5625 - 10*1600
            fo.close()
            if Plot:
                tmp = delay[2,:]
                plt.hist(tmp, np.linspace(-5000,5000,201))
                plt.ylim([0,lim_max])
                plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_delay.jpg')
                plt.clf()
            #### unique LOR id
            coin_LOR =np.int64(data[0,:])*13824 +  data[1,:]
            delay_LOR =np.int64(delay[0,:])*13824 +  delay[1,:]
            del_pos = np.array([])
            coin = np.zeros((5,data.shape[1]+delay.shape[1]), dtype = np.int16)
            #### coin events filled
            coin[0,:data.shape[1]] = coin_LOR
            #### coin id
            coin[1,:data.shape[1]] = 0
            coin[2:5,:data.shape[1]] = data
            
            #### delay events filled
            coin[0,data.shape[1]:] = delay_LOR
            #### delay id
            coin[1,data.shape[1]:] = 1
            coin[2:5,data.shape[1]:] = delay
            
            #### unique event id
            index = np.linspace(0,data.shape[1]+delay.shape[1]-1,data.shape[1]+delay.shape[1], dtype = np.int64)
            #### sort based on coin/delay id << time << LOR id
            argsort = np.lexsort((coin[1,:], coin[4,:], coin[0,:]))
            coin_sorted = coin[:,argsort]
            index = index[argsort]
            prev = 0
            
            
            #### subtract coin events which have same nearby delay events with same LOR id.
            for i in range(1000):
                #### same LOR crystal pair but one coin and one delay
                valid = np.insert((np.diff(coin_sorted[0,]) == 0)&(np.diff(coin_sorted[1,]) == 1),0,False)
                ###               --- within same LOR        ---  and next is delay but ours is coin ---
                valid2 = (coin_sorted[1,:] == 1) # is a delay
                curr = np.sum(valid2) # number of delays
                print("residual delay: ", curr) 
                if(curr == prev):
                    index = index[~valid2] # gets rid of remaining delays
                    break
                prev = curr 
                valid = valid | (np.insert(valid[1:],valid.size - 1,False)) # insert false at end
                index = index[~valid]
                coin_sorted = coin_sorted[:,~valid]
            coin = coin[:,index]
            data = coin[2:5,:]
            f_random = open(f_random_corrected, 'wb')
            #### .write doesn't need transpose function
            f_random.write(data.tobytes())
            f_random.close()
        else:
            fo=open(f_random_corrected,"rb")
            data2 = np.fromfile(fo,dtype = np.int16)
            data2 = np.reshape(data2, (3, int(data2.shape[0]/3)))
            data = data2
            fo.close()
        if Plot:
            tmp = data[2,:]
            plt.hist(tmp, np.linspace(-5000,5000,201))
            plt.ylim([0,lim_max])
            plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected.jpg')
            plt.clf()
       
    
    #### find LOR skew lut table
    if Skew:
        f_skew = result_path + '/' + str(sub0) + '_' + str(sub1) + '_skew_array.dat'
        f_skew2 = 'skew_lut.dat'
        if os.path.isfile(f_skew):
            skewfile = open(f_skew, 'rb')
            skewoffset = np.fromfile(skewfile, np.int16)
            skewoffset = np.reshape(skewoffset, (pixel_num, pixel_num))
        elif os.path.isfile(f_skew2):
            data_all = open(f_skew2, 'rb')
            data_all = np.fromfile(data_all, np.int16)
            data_all = np.reshape(data_all, (864*16, 864*16))
            skewoffset = data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num]
            print('loading skewoffset')
        else:
            #### split data based on crystalid pair (LOR)
            data_argsort = np.lexsort((data[1, :], data[0, :]))
            data_sorted = data[:, data_argsort]
            data_split_pos = np.where(np.diff(data_sorted[1, :]))[0] + 1
            data_unique_crystal1 = np.int16(data_sorted[0, np.insert(data_split_pos, 0, 0)])
            data_unique_crystal2 = np.int16(data_sorted[1, np.insert(data_split_pos, 0, 0)])
            data_split = np.split(data_sorted[2, :], data_split_pos)
            
            #### find the max count for all LORs and which LOR has the max count
            max_len = np.max([i.shape[0] for i in data_split])
            max_len_i = np.argmax([i.shape[0] for i in data_split])
            frag_size = 2000000 * max((1, int(1000/max_len)))
            num = int(len(data_split)/ frag_size)
            print(len(data_split), max_len, frag_size)
            for i in range(num + 1):
                #### make all LOR have same counts. Less count LORs are filled with NAN
                data_aranged = stack_padding(data_split[i*frag_size: np.minimum(len(data_split),(i+1)*frag_size)])
                #### find the mean value
                offset = np.nanmean(data_aranged, axis = 1)
                
                #### find the mean value of the right Gaussian [offset, offset + 2000]
                l_b = np.transpose(np.repeat([offset], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged2 = np.array(data_aranged)
                data_aranged2[(data_aranged < l_b)|(data_aranged > r_b)] = np.nan
                offset2 = np.nanmean(data_aranged2, axis = 1)
                
                #### find the mean value of the left Gaussian [offset - 2000, offset]
                l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset], data_aranged.shape[1] ,axis = 0))
                data_aranged3 = np.array(data_aranged)
                data_aranged3[(data_aranged < l_b)|(data_aranged > r_b)] = np.nan
                offset3 = np.nanmean(data_aranged3, axis = 1)
               
                #### skew is defined as the average of the left Gaussian center and the right Gaussian center
                offset4 = (offset3 + offset2) * 0.5
                skewoffset[data_unique_crystal1[i*frag_size: (i+1)*frag_size]%pixel_num,data_unique_crystal2[i*frag_size: (i+1)*frag_size]%pixel_num] = offset4
                
                #### store skew_lut to file
                skewfile = open(f_skew, 'wb')
                skewfile.write(skewoffset.tobytes())
                skewfile.close()
                
                
                #### make all LOR have same counts. Less count LORs are filled with NAN
                data_aranged = stack_padding(data_split[i*frag_size: np.minimum(len(data_split),(i+1)*frag_size)])
                #### find the mean value
                offset = np.nanmean(data_aranged, axis = 1)
                
                #### find the mean value with bound [offset - 2000, offset + 2000]
                l_b = np.transpose(np.repeat([offset - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged0 = np.array(data_aranged)
                data_aranged0[(data_aranged0 < l_b)|(data_aranged0 > r_b)] = np.nan
                offset0 = np.nanmean(data_aranged0, axis = 1)
                
                
                #### find the mean value of the right Gaussian [offset0, offset0 + 2000]
                l_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0 + 2000], data_aranged.shape[1] ,axis = 0))
                data_aranged2 = np.array(data_aranged)
                data_aranged2[(data_aranged2 < l_b)|(data_aranged2 > r_b)] = np.nan
                offset2 = np.nanmean(data_aranged2, axis = 1)
                
                #### find the mean value of the left Gaussian [offset0 - 2000, offset0]
                l_b = np.transpose(np.repeat([offset0 - 2000], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset0], data_aranged.shape[1] ,axis = 0))
                data_aranged3 = np.array(data_aranged)
                data_aranged3[(data_aranged3 < l_b)|(data_aranged3 > r_b)] = np.nan
                offset3 = np.nanmean(data_aranged3, axis = 1)
                
                #### find the mean value of the right Gaussian [offset2 - 500, offset2 + 500]
                l_b = np.transpose(np.repeat([offset2-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset2+500], data_aranged.shape[1] ,axis = 0))
                data_aranged4 = np.array(data_aranged)
                data_aranged4[(data_aranged4 < l_b)|(data_aranged4 > r_b)] = np.nan
                offset4 = np.nanmean(data_aranged4, axis = 1)
                
                #### find the mean value of the left Gaussian [offset3 - 500, offset3 + 500]
                l_b = np.transpose(np.repeat([offset3-500], data_aranged.shape[1] ,axis = 0))
                r_b = np.transpose(np.repeat([offset3+500], data_aranged.shape[1] ,axis = 0))
                data_aranged5 = np.array(data_aranged)
                data_aranged5[(data_aranged5 < l_b)|(data_aranged5 > r_b)] = np.nan
                offset5 = np.nanmean(data_aranged5, axis = 1)
                
                offset8 = (offset4 + offset5) * 0.5
                skewoffset[data_unique_crystal1[i*frag_size: (i+1)*frag_size]%pixel_num,data_unique_crystal2[i*frag_size: (i+1)*frag_size]%pixel_num] = offset8
        data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num] = skewoffset
        if Plot:
            tmp = data[2,:] - data_all[np.uint16(data[0,:]),np.uint16(data[1,:])]
            plt.hist(tmp, np.linspace(-5000,5000,201))
            plt.ylim([0,lim_max])
            plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected_tof.jpg')
            plt.clf()
    
    
    #### output listmode data
    if Listmode:
        f_listmode = listmode_path + '/' + str(sub0) + '_' + str(sub1) + '.lm'
        if not os.path.isfile(f_listmode):
            numberOfCoincidenceEvents = data.shape[1]
            listmodedata = np.zeros((numberOfCoincidenceEvents,10),dtype = np.float32)
            print('listmode data shape',listmodedata.shape)
            #### get crystal1 position xyz
            crystalID1_arr = np.uint16(data[0,:])
            tmp = crystalID1_arr
            tmp = Det_convert[np.uint16(tmp/864)] * 864 + tmp%864
            crystalID1_arr_convert = tmp 
            listmodedata[:,0] = crystalPositionMap[tmp,0] # firstCrystalX
            listmodedata[:,1] = crystalPositionMap[tmp,1] # firstCrystalY
            listmodedata[:,2] = crystalPositionMap[tmp,2] # firstCrystalZ

            #### get crystal2 position xyz
            crystalID2_arr = np.uint16(data[1,:])
            tmp = crystalID2_arr
            tmp = Det_convert[np.uint16(tmp/864)] * 864 + tmp%864
            crystalID2_arr_convert = tmp 
            listmodedata[:,5] = crystalPositionMap[tmp,0] # SecondCrystalX
            listmodedata[:,6] = crystalPositionMap[tmp,1] # SecondCrystalY
            listmodedata[:,7] = crystalPositionMap[tmp,2] # SecondCrystalZ
            
            #### get TOF info
            if TOF:
                offset_arr = data_all[crystalID1_arr,crystalID2_arr]
                timediff_ps = np.float32(offset_arr) - np.float32(data[2,:])
                TOF_arr = speedOfLight_length_ps * timediff_ps
                listmodedata[:,3] = TOF_arr # TOF
                if Plot:
                    tmp = TOF_arr
                    plt.hist(tmp, np.linspace(-600,600,25))
                    plt.ylim([0,lim_max])
                    plt.savefig(result_path + '/' + str(sub0) + '_' + str(sub1) + '_coin_random_corrected_tof_listmode.jpg')
                    plt.clf()
            print("total coins:", numberOfCoincidenceEvents)
            
            
            #### store all coin events to listmode data
            l_index = np.arange(numberOfCoincidenceEvents)
            np.random.shuffle(l_index)
            listmodedata = listmodedata[l_index, :]
            with open(f_listmode, 'wb') as lm:  
                seg_len = 100000
                for k in range(int(numberOfCoincidenceEvents/seg_len)):
                    lm.write(listmodedata[k*seg_len:(k+1)*seg_len,:].tobytes())
                k = int(numberOfCoincidenceEvents/seg_len)
                lm.write(listmodedata[k*seg_len:,:].tobytes())
            lm.close()
            
            
            #### save a fraction data to txt file for validation
            eventdata = np.zeros((100,15),dtype = np.float32)
            eventdata[:,:10] = listmodedata[:100,:]
            eventdata[:,10] = 0
            eventdata[:,11] = crystalID1_arr[:100]
            eventdata[:,12] = crystalID2_arr[:100]
            eventdata[:,13] = crystalID1_arr_convert[:100]
            eventdata[:,14] = crystalID2_arr_convert[:100]
            with open(f_listmode.replace('lm','txt'), 'a') as f:
                np.savetxt(f, eventdata, fmt=textOutputFmt)

#### combine all detid pair skew_lut data to a big file
if Skew:    
    Files = os.listdir(result_path)
    Files = [i for i in Files if '.dat' in i and 'skew' in i and 'lut' not in i]
    Files.sort(key=lambda f: int(f.split('_')[0])*1000 + int(f.split('_')[1]))
    print(Files)
    for f in Files:
        print(f)
        sub0 = int(f.split('_')[0])
        sub1 = int(f.split('_')[1])
        skewfile = open(result_path + '/' +  f, 'rb')
        data3 = np.fromfile(skewfile, np.int16)
        data3 = np.reshape(data3, (pixel_num, pixel_num))
        data_all[sub0*pixel_num:sub0*pixel_num+ pixel_num, sub1*pixel_num:sub1*pixel_num+ pixel_num] = data3
    f_skew = result_path + '/' +  'skew_lut.dat'
    skewfile = open(f_skew, 'wb')
    skewfile.write(data_all.tobytes())
    skewfile.close()
    
   
#### listmode data per detid pair based needs to be resampled and shuffled and then combined.
if ListmodeCombine:
    lstFiles = os.listdir(listmode_path)
    lstFiles = [i for i in lstFiles if '.lm' in i and '_' in i]
    print(lstFiles)
    o_dir = listmode_path.replace('\\','/').split('/')[-3]
    f_out = listmode_path + '/' + o_dir +'.lm'
    if not (os.path.isfile(f_out)):
        file_dict = {}
        fext=open(f_out,"wb")
        for i in range(1001):
            if i%100 == 0:
                print(i)
            coin_num = 0
            listmodedata = np.zeros((0,10), dtype = np.float32)
            for f in lstFiles:
                if f not in file_dict.keys():
                    file_dict[f]=[open(os.path.join(listmode_path,f),"rb"), os.path.getsize(os.path.join(listmode_path,f)), 0] 
                counts = np.int32(file_dict[f][1] / 40 / 1000) * 10
                if i == 1000:
                    counts = np.int32((file_dict[f][1] - file_dict[f][2]) / 4)
                data = np.fromfile(file_dict[f][0],dtype = np.float32, count= counts)
                file_dict[f][2] += data.shape[0]*4
                data = np.reshape(data, (np.int32(data.shape[0]/10),10))
                coin_num += np.int32(data.shape[0])
                listmodedata = np.concatenate([listmodedata,data],axis = 0)
                if i == 1000:
                    file_dict[f][0].close()
            index = np.arange(coin_num)
            np.random.shuffle(index)
            fext.write(listmodedata[index, : ].tobytes())
        fext.close()
