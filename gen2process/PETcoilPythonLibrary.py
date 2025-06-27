from scipy import optimize  # Scipy's package for curve fitting and optimization
from array import array
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import os
import sys
import time
from sklearn.metrics import r2_score
from lmfit.models import ExponentialModel, GaussianModel
import pickle
import copy
from scipy.interpolate import interp1d
import math
import struct

numOfBitsForEnergy = 9
numOfBinsInHistogram = 512
numOfTDCbins = 32
numOfCrystals = 13824
coarseTimeInPicoSec = 1600  # unit in ps: 1600 for 625 MHz, 1462 for 684 MHz
fineTimeInPicoSec = 50
timeSpectrumWidth = 3  # in coarse time
numOfBinsInTimeSpectrum = timeSpectrumWidth*2*32
TDCbinwidth =1024
newsub_break_pos = 1000
SMALL_SIZE = 16
MEDIUM_SIZE = 18
LARGE_SIZE = 16
speedOfLight = 299792458000  # speed of light in mm
speedOfLight_length_ps = speedOfLight * math.pow(10, -12)


# convert hexadecimal to binary
def hextobin(byte):
    bytehex = byte.hex()
    scale = 16  # equals to hexadecimal
    data_size = len(bytehex)*4
    return (bin(int(bytehex, scale))[2:].zfill(data_size))

def reportProgress(currentPos, numOfBytesInFile):
        print(currentPos / numOfBytesInFile * 100, '%', ' completed')


def GetPhotopeaksPositionCrystal(crystalID, energyPerCrystal, plot=False):
    # find peak position
    energyPerCrystal_fit = energyPerCrystal
    if(np.max(energyPerCrystal_fit[0:10]) >= np.max(energyPerCrystal_fit)):
        energyPerCrystal_fit[:50] = 0
        print(crystalID, 'cut before 511 peak')
    peaksIndices = peakutils.indexes(energyPerCrystal_fit, thres=0.013, min_dist=15)
    peaksIndices = peaksIndices[peaksIndices > peaksIndices[-1]/2]
    #plt.plot(energyPerCrystal)
    #plt.plot(peaksIndices,energyPerCrystal[peaksIndices],'r*')
    #print(peaksIndices)
    #plt.show()
    countOfPeaks = energyPerCrystal[peaksIndices]
    peakIndexWithMaxCount = np.argmax(countOfPeaks)
    firstnonzeroindex = np.nonzero(energyPerCrystal)[0][0]
    Pos511 = peakIndexWithMaxCount
    if peakIndexWithMaxCount < len(countOfPeaks) - 1:
        peakIndexWithSecondMaxCount = np.argmax(countOfPeaks[peakIndexWithMaxCount + 1:]) + peakIndexWithMaxCount + 1
        if countOfPeaks[peakIndexWithMaxCount] < countOfPeaks[peakIndexWithSecondMaxCount] * 1.5:
            if(peakIndexWithSecondMaxCount > peakIndexWithMaxCount):
                Pos511 = peakIndexWithSecondMaxCount
    #print(peaksIndices[Pos511],firstnonzeroindex, peakIndexWithMaxCount, peakIndexWithSecondMaxCount, peaksIndices,countOfPeaks,energyPerCrystal)
    
    # find photopeaks position, assuming 511 keV photopeak has the most
    # counts and 1.27 MeV photopeak has the largest index
    photoPeaksIndex = [peaksIndices[Pos511], peaksIndices[-1]]
    # plot the energy spectrum and the photopeaks found for checking
    if plot is True:
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.plot(energyPerCrystal)
        plt.plot(photoPeaksIndex, energyPerCrystal[photoPeaksIndex], '.')
        plt.title('crystal: ' + str(crystalID))
        plt.show()
    return photoPeaksIndex


# exponential function that models non-linear energy response of SiPM
def keVtoADC(keV, a, b):
    return a * (1 - np.exp(-b * keV))

# convert the ADC channel back to keV (linearize energy spectrum)
def ADCtokeV(ADC, a, b):
    tmp = 1 - (ADC / a)
    tmp[tmp < 0] = 1
    return -(1 / b) * np.log (tmp)

# linearize energy spectrum
def GetLinearizedEnergySpectrum( energyPerCrystal, params = [0,0], crystalID = 0, plot=False):
    photoPeaksIndex = [0, 0]
    if np.sum(params) == 0:
        photoPeaksIndex = GetPhotopeaksPositionCrystal(crystalID, energyPerCrystal)
        isotopePhotopeaks = [0., 511., 1274.5] # Na-22: 511 and 1274.5 keV, added 0 energy for better fitting
        ADCvalue = [0] + photoPeaksIndex
        initialGuessForFitting = [100, 0.002]
        try: 
            params, params_covar = optimize.curve_fit(keVtoADC, isotopePhotopeaks, ADCvalue, p0=initialGuessForFitting)
        except RuntimeError:
            return 0,0,photoPeaksIndex,0,True
    # convert ADC to keV to plot linearize energy spectrum
    energySpectrumCount = energyPerCrystal
    x = np.linspace(0, len(energySpectrumCount)-1, len(energySpectrumCount))
    #print(params)
    keV = ADCtokeV(x, params[0], params[1])
    ADC = keVtoADC(np.linspace(0, 1300-1, 1300), params[0], params[1])
    
    if plot is True:
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        # plot the exponential fit for checking
        plt.plot(isotopePhotopeaks, ADCvalue, '.')
        plt.plot(keV, keVtoADC(keV, params[0], params[1]))
        plt.show()

        # plot the linearized energySpectrum for checking
        plt.plot(keV, energySpectrumCount, '-')
        plt.show()
    
    params = params[0:2]
    
    return keV, ADC, photoPeaksIndex, params, False


def GetEnergyResolution(energyPerCrystal, coincidencePair, plot=False, dir=0, 
                        Onlycoincidence = False, font_size = LARGE_SIZE, 
                        energyWindow = [450,600], clean = False,paramsPerCrystal='', 
                        Exponentialfit = True, interpret = 1, Globalonly = False, photoPeakBoundaryPerCrystal = ''):
    energySpectrum_twopeaks = copy.deepcopy(energyPerCrystal)
    plt.figure()
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.figsize"] = (10,8)
    if Onlycoincidence:
        print(coincidencePair.shape)
        energyPerCrystal = {}
        # filter out coincidences from the same submodule
        index_valid = np.floor(coincidencePair[4, :] / 144) != np.floor(coincidencePair[5, :]/144)
        coincidencePair = coincidencePair[:,index_valid]
        crystal_id = np.concatenate((coincidencePair[4, :], coincidencePair[5, :]), axis=0)
        energy = np.concatenate((coincidencePair[6,:],coincidencePair[7,:]),axis=0)
        crystalid_argsort = np.argsort(crystal_id)
        crystalid_sorted = crystal_id[crystalid_argsort]
        crystalid_split_pos = np.where(np.diff(crystalid_sorted))[0] + 1
        energy_split = np.split(energy[crystalid_argsort], crystalid_split_pos)
        crystalid_unique = crystalid_sorted[np.insert(crystalid_split_pos, 0, 0)]
        for i in range(crystalid_unique.size):
            crystalid = crystalid_unique[i]
            if crystalid not in energyPerCrystal:
                energyPerCrystal[crystalid] = np.zeros(512, dtype=np.int64)
            energy_sorted = np.sort(energy_split[i])
            energy_split_pos = np.where(np.diff(energy_sorted))[0] + 1
            energy_unique_pos = np.insert(energy_split_pos, [0, energy_split_pos.size], [0, energy_sorted.size])
            energy_sum_perbin = np.diff(energy_unique_pos)

            energy_value_perbin = energy_sorted[energy_unique_pos[:-1]]
            energyPerCrystal[crystalid][energy_value_perbin] = energy_sum_perbin
    if not os.path.exists(dir):
        os.makedirs(dir) 
    
    global_energy = np.zeros(700)
    energyResolutionPerCrystal = {}
    
    if photoPeakBoundaryPerCrystal == '':
        photoPeakBoundaryPerCrystal = np.zeros((numOfCrystals, 2), dtype=np.uint32)
    if len(paramsPerCrystal) == 0:
        paramsPerCrystal = np.zeros((numOfCrystals, 2), dtype=np.float64)
    params_valid = np.sum(paramsPerCrystal) > 0
    photoPeaksIndexPerCrystal = np.zeros((numOfCrystals, 2), dtype=np.float64)
    invalid_number = 0
    global_single_count_gated = 0
    if Globalonly:
        for crystal in list(energySpectrum_twopeaks):
            global_single_count_gated += \
                np.sum(energySpectrum_twopeaks[crystal][photoPeakBoundaryPerCrystal[crystal, 0]:photoPeakBoundaryPerCrystal[crystal, 1]])
    else:
        for crystal in list(energyPerCrystal):
        #for crystal in  [1820, ]:
            invalid = False
            energySpectrum = energyPerCrystal[crystal]
            #if not np.count_nonzero(energySpectrum[100:] > 50) > 3:
            if not np.count_nonzero(energySpectrum > 0) > 10:
                invalid = True
                invalid_number = 1
                print(crystal, ": NA ENERGY PLOT")
            if np.int64(crystal/144) == newsub_break_pos:
                invalid = True
                invalid_number = 2
                print("skip this submodule")
            filename = dir + '/' + str(crystal)
            if(crystal>numOfCrystals-1):
                print(crystal)
                print(energyPerCrystal.keys())
            params = paramsPerCrystal[crystal, :]
            if((np.sum(params) == 0) & params_valid):
                invalid = 1
            energyResolution_result, photoPeakBoundary, energySpectrumCount_filled, params,  photoPeaksIndex\
                = GetEnergyResolutionforcrystal(energySpectrum_twopeaks[crystal], energySpectrum, invalid = invalid, 
                                                invalid_number = invalid_number, plot=plot, filename=filename, 
                                                Onlycoincidence = Onlycoincidence, font_size = font_size, energyWindow = energyWindow, 
                                                clean = clean, Global = False, params = params, Exponentialfit = Exponentialfit, 
                                                interpret = interpret)
            paramsPerCrystal[crystal, :] = params
            photoPeakBoundaryPerCrystal[crystal, :] = photoPeakBoundary
            photoPeaksIndexPerCrystal[crystal, :] = photoPeaksIndex
            
            global_single_count_gated += np.sum(energySpectrum_twopeaks[crystal][photoPeakBoundary[0]:photoPeakBoundary[1]])
            if not invalid:
                global_energy += energySpectrumCount_filled
                energyResolutionPerCrystal[crystal] = energyResolution_result
            
            
            if invalid:
                if crystal in energyPerCrystal.keys():
                    del energyPerCrystal[crystal]
    print("global")
    filename = os.path.split(dir)[0] + '/Global_energy'
    energyResolution_result, photoPeakBoundary, energySpectrumCount_filled, params, photoPeaksIndex = \
        GetEnergyResolutionforcrystal(global_energy, global_energy , invalid = 0, invalid_number = 0, 
            plot=plot, filename=filename, Onlycoincidence = Onlycoincidence, font_size = font_size, 
            energyWindow = energyWindow, clean = clean, Global = True, Exponentialfit = Exponentialfit, interpret = interpret)   
    global_energy_resolution = energyResolution_result[0]
    plt.clf()
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.figsize"] = (10,8)
    with open(os.path.split(dir)[0] + '/paramsPerCrystal.pickle', 'wb') as file:
        pickle.dump(paramsPerCrystal, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.split(dir)[0] + '/photoPeakBoundaryPerCrystal.pickle', 'wb') as file:
        pickle.dump(photoPeakBoundaryPerCrystal, file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.split(dir)[0] + '/photoPeaksIndexPerCrystal.pickle', 'wb') as file:
        pickle.dump(photoPeaksIndexPerCrystal, file, pickle.HIGHEST_PROTOCOL)
    return [energyResolutionPerCrystal,photoPeakBoundaryPerCrystal], [global_single_count_gated, global_energy_resolution]

def GetEnergyResolutionforcrystal(energySpectrum_twopeaks, energySpectrum, invalid = False, invalid_number = 0, plot=False, filename=0, Onlycoincidence = False, font_size = LARGE_SIZE, energyWindow = [450,600], clean = False, Global = False, params = [0,0],Exponentialfit = True, interpret = 1):
     
    energySpectrumCount_filled = np.zeros(700) 
    photoPeakBoundary = [0, 0]
    energyResolution_result = [0, 0, 0]
    energySpectrum_copy = energySpectrum
    photoPeaksIndex = [0,0]
    if Global:    
        #gloab energy curve fitting with exponential + gaussian.
       # with open(filename+'global.pickle', 'rb') as file:
       #     energySpectrum = pickle.load(file)
        energySpectrum = energySpectrum / 1000
        energySpectrum_copy = energySpectrum
        fit_y = energySpectrum[energyWindow[0]:energyWindow[1]]
        fit_y_o = fit_y
        fit_x = np.linspace(energyWindow[0], energyWindow[1]-1, energyWindow[1]-energyWindow[0])
        keV = np.linspace(0, 700-1, 700)
    else:     
        if not invalid:
            photoPeakBoundary, params, invalid, invalid_number, keV, ADC, photoPeaksIndex = GetPhotopeaksparams(energySpectrum_twopeaks, energyWindow, params)
            energyPerCrystal_bounded = energySpectrum_twopeaks[photoPeakBoundary[0]:photoPeakBoundary[1]]
            if (photoPeakBoundary[1] - photoPeakBoundary[0] < 5):
                invalid = True
                invalid_number = 2
            
        if not invalid:             
            # limit keV' range: (0,700]
            keV = np.round(keV).astype(np.float64)
            valid = (keV < 700) & (keV > 0)
            keV = keV[valid, ]
            energySpectrum = energySpectrum[valid, ] 
            
                
        if not invalid:
            # fill the energySpectrum with all 1keV filled. Split one point in the spectrum to all nearby points.
            keV_filled = [i for i in range(np.round(keV[0]).astype(np.int64),np.round(keV[-1]).astype(np.int64)+1)]
            # find nonzero points
            energySpectrumCount_nonzero_index = np.where(np.array(energySpectrum) != 0)[0]
            energySpectrumCount_nonzero= energySpectrum[energySpectrumCount_nonzero_index]
            keV_nonzero = keV[energySpectrumCount_nonzero_index]
            keV_nonzero_middle = np.int32(np.round((keV_nonzero[1:] + keV_nonzero[:-1])/2))
            f = interp1d(keV_nonzero, energySpectrum[energySpectrumCount_nonzero_index])
            index = np.linspace(keV_nonzero[0].astype(int), keV_nonzero[-1].astype(int), keV_nonzero[-1].astype(int) - keV_nonzero[0].astype(int) + 1).astype(int)
            energySpectrumCount_filled[index] = f(index) / np.sum(f(index))*np.sum(energySpectrum)
            keV = np.linspace(0, 700-1, 700)
            if interpret == 1:
                energySpectrum = energySpectrumCount_filled
            else:
                energySpectrumCount_filled2 = np.zeros(700) 
                for i in range(len(keV_nonzero_middle)-1):
                    energySpectrumCount_filled2[keV_nonzero_middle[i]:keV_nonzero_middle[i+1]] = energySpectrumCount_nonzero[i+1]
                energySpectrumCount_filled2 =energySpectrumCount_filled2 / np.sum(energySpectrumCount_filled2)*np.sum(energySpectrum)
                energySpectrum = energySpectrumCount_filled
                energySpectrumCount_filled = energySpectrumCount_filled2
            #print(energySpectrumCount_filled)
            #plt.plot(energySpectrumCount_filled2)
            #plt.plot(energySpectrum)
            #plt.show()
            # constrain the range to fit around the photopeak
            expectedEnergyResolution = 0.13
            #fitFrom511_keV = keV[ photoPeaksIndex[0]] - np.round(keV[photoPeaksIndex[0]] * 0.4 * 2 * expectedEnergyResolution).astype(np.int)
            #fitTo511_keV = keV[photoPeaksIndex[0]] + np.round(keV[photoPeaksIndex[0]] * 0.6 * 2 * expectedEnergyResolution).astype(np.int)
            fit_x = keV[energyWindow[0]:energyWindow[1]]
            fit_y = energySpectrum[energyWindow[0]:energyWindow[1]]
            if len(fit_x) < 10:
                invalid = True  
                invalid_number = 7
    if not invalid:    
        try:
            parametersGaussian = peakutils.gaussian_fit(fit_x,fit_y , center_only=False)
        except RuntimeError:
            invalid = True  
            invalid_number = 8
        except ValueError:
            invalid = True  
            invalid_number = 8
    
    if not invalid:      
        # add gaussian model to fit
        gauss1 = GaussianModel(prefix='g1_')
        pars = gauss1.guess(fit_x, fit_y)
        pars.update(gauss1.make_params())
        tmp = 511
        ratio = 0.3
        pars['g1_center'].set(value=tmp, min = tmp*(1-ratio), max = tmp*(1+ratio))
        tmp = 25
        ratio = 0.3
        pars['g1_sigma'].set(value=tmp, min = tmp*(1-ratio), max = tmp*(1+ratio))
        tmp = np.max(fit_y)
        ratio = 0.3
        pars['g1_height'].set(value=tmp, min = tmp*(1-ratio), max = tmp*(1+ratio))  
        mod = gauss1
        if Exponentialfit:
            # fit background radiation
            exp_mod = ExponentialModel(prefix='exp_')
            # use curves before and after 511 peak  
            t = [380,420,620,680]
            # find the cutting start point
            tmp = np.where(energySpectrum > 0)[0][0]
            # bound the four dots in [cutting start point, length of keV(700 keV)]
            # if the first two dots are equal ( = cutting start point), force the second point +5
            if t[0] == t[1]:
                t[1] = t[1] + 10
            # fit (x,y) on the two curves
            x_efit = np.concatenate([np.array(keV[t[0]:t[1]]),np.array(keV[t[2]:t[3]])])
            y_efit = np.concatenate([np.array(energySpectrum[t[0]:t[1]]),np.array(energySpectrum[t[2]:t[3]])])
            pars_e = exp_mod.guess(np.array(keV[t]), np.array(energySpectrum[t]))
            #pars['exp_decay'].set(min = 120, max = 320)
            # if fit is not successful, then use guess paramters. Compare fit the whole spectrum or the two curves (before and after 511), choose the best fitting result
            try:
                out1 = exp_mod.fit(energySpectrum[t], pars_e, x=keV[t])
                pars1 = out1.params
            except:
                pars1 = pars_e
            try:
                out2 = exp_mod.fit(y_efit, pars_e, x=x_efit)
                pars2 = out2.params
            except:
                pars2 = pars_e
            r2_1 = r2_score(y_efit, exp_mod.eval(pars1, x= x_efit))
            r2_2 = r2_score(y_efit, exp_mod.eval(pars2, x= x_efit))
            if r2_2 > r2_1:
                pars_e = pars2
            else:
                pars_e = pars1
            pars_e['exp_amplitude'].set(vary = False)
            pars_e['exp_decay'].set(vary = False)
            fit_result = exp_mod.eval(pars_e, x= x_efit)
            bg_pars = pars_e
            r2 = r2_score(y_efit, fit_result)
            if (r2 < 0.9):
        
                plt.rcParams.update({'font.size': font_size})
                plt.rcParams["figure.figsize"] = (10,8)
                plt.plot(keV, energySpectrum, 'k-', label='Experiment')
                plt.plot(x_efit, fit_result, 'r--',label =r'All Fit (R^2 = '+str(np.round(r2,3))+')')
                r2 = r2_score(y_efit, fit_result)
                plt.plot(x_efit, fit_result, 'g--',label =r'BG Fit (R^2 = '+str(np.round(r2,3))+')')
                plt.plot(x_efit, y_efit, 'm*')
                plt.axvline(x=keV[photoPeakBoundary[0]], color='r', linestyle='-')
                plt.axvline(x=keV[photoPeakBoundary[1]], color='r', linestyle='-')
                plt.plot(keV[t], energySpectrum[t], 'r*')
                plt.title(str(invalid_number) + ': ' + 'bad data')
                plt.legend(loc='upper right')
                plt.savefig(filename +'_'+str(10)+'_bad.png')
                plt.clf()
                plt.rcParams.update({'font.size': font_size})
                plt.rcParams["figure.figsize"] = (10,8)
            #pars.update(pars_e)
            #mod = gauss1 + exp_mod
        try:
            if Exponentialfit:
                bg = exp_mod.eval(pars_e, x= fit_x)
                fit_y -= bg
                fit_leftbound = int(len(fit_y)*0)
                fit_rightbound = int(len(fit_y)*1)
            else:
                fit_leftbound = int(len(fit_y)*0.3)
                fit_rightbound = int(len(fit_y)*0.7)
                
            out = mod.fit(fit_y[fit_leftbound:fit_rightbound], pars, x=fit_x[fit_leftbound:fit_rightbound])
            r2 = r2_score(fit_y[fit_leftbound:fit_rightbound], out.best_fit)
        except ValueError:
            invalid = True  
            invalid_number = 9
            #print(filename, invalid_number)
       
    # if('864' in filename):
        # print(invalid,plot)
        # plt.plot(energySpectrum)
        # plt.plot(fit_x, fit_y)
        # plt.plot(fit_x[int(fit_x.size/4):], fit_y[int(fit_x.size/4):])
        # plt.plot(energySpectrum_twopeaks)
        # plt.axvline(x=photoPeakBoundary[0], color='r', linestyle='-')
        # plt.axvline(x=photoPeakBoundary[1], color='r', linestyle='-')
        # plt.show()      
      
    if not invalid:
        test_string = out.fit_report()
        t = test_string.split()
        if clean:
            for i in range(2, len(t)):
                if t[i-1] == 'g1_fwhm:':
                    print("DEBUG7: ",t[i],t[i-1])
                if t[i-1] == 'g1_center:':
                    print("DEBUG8: ",t[i],t[i-1])
        g1_fwhm = [float(t[i]) for i in range(2,len(t)) if t[i-1] == 'g1_fwhm:']
        g1_height = [float(t[i]) for i in range(2,len(t)) if t[i-1] == 'g1_height:']
        g1_center = [float(t[i]) for i in range(2,len(t)) if t[i-1] == 'g1_center:' and t[i] != 'at']
        
        if (g1_center[0] == 0):
            print("g1_center[0] == 0")
            energyResolution = 0
        else:
            energyResolution = abs(g1_fwhm[0] /g1_center[0])
        if Global:
            energyResolution_result = [energyResolution, np.sum(energySpectrum*1000), np.sum(fit_y*1000).astype(np.int64)]
        else:
            energyResolution_result = [energyResolution, np.sum(energySpectrumCount_filled), np.sum(energyPerCrystal_bounded)]
        
        if r2 < 0.75:
            invalid = True
            invalid_number = 6
            
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.figsize"] = (10,8)
    if invalid:   
        if invalid_number != 1:
            if invalid_number == 6:
        
                plt.rcParams.update({'font.size': font_size})
                plt.rcParams["figure.figsize"] = (10,8)
                plt.plot(keV,energySpectrum)
                plt.plot(fit_x, fit_y, 'b-', label='Experiment')
                plt.plot(fit_x, out.eval(out.params,x = fit_x), 'r--',label =r'Fit (R^2 = '+str(np.round(r2,3))+')')
            if invalid_number > 2:
                plt.axvline(x=photoPeakBoundary[0], color='r', linestyle='-')
                plt.axvline(x=photoPeakBoundary[1], color='r', linestyle='-')
        
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)
            plt.plot(energySpectrum_copy,'k-',label='Raw')
            plt.legend(loc='upper right')
            plt.title(str(invalid_number) + ': ' + 'bad data')
            plt.tick_params(axis='x', rotation=45)
            plt.xlabel('Energy (keV)')
            plt.ylabel('Count (cnts)')
            plt.savefig(filename +'_'+str(invalid_number)+'_bad.png')
            plt.clf()
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)
        
    
    elif Global:
        print('Energy Global')
        components = out.eval_components(x=fit_x)
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.plot(fit_x, fit_y_o, 'b-', label='Experiment')
        if Exponentialfit:
            plt.plot(fit_x, fit_y, 'r--',label ='Without BG')
            plt.plot(fit_x, bg, 'g-.',label ='BG Fit')
        plt.plot(fit_x, components['g1_'], 'm:',label ='Gaussian Fit \nR^2 = '+str(np.round(r2,3)))
        #plt.plot(keV,energySpectrum/1000)
        plt.legend(loc='upper right')
        plt.title('FWHM: ' + str(np.round(energyResolution*100, 1)) + '%')
        plt.xlim(energyWindow[0], energyWindow[1])
        plt.xlabel('Energy (keV)')
        plt.ylabel('Count (kcnts)')
        plt.savefig(filename+'.png')
        plt.clf()
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        
    elif plot:    
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)    
        plt.plot(fit_x, fit_y, 'b-', label='Experiment')
        plt.plot(fit_x, out.eval(out.params,x = fit_x), 'r--',label =r'Fit (R^2 = '+str(np.round(r2,3))+')')
        #plt.plot(keV,energySpectrum/1000)
        plt.legend(loc='upper right')
        plt.title('FWHM:' + str(np.round(energyResolution*100, 2)) + '%')
        #plt.xlim(450, 600)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Count (cnts)')
        plt.savefig(filename+'.png')
        plt.clf()
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        
     
         
    if Global:
        try:
        
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)
            plt.plot(fit_x, fit_y, 'b-', label='Experiment')
            plt.plot(keV[350:], out1.eval(exp_mod.params,x = keV[350:]), 'k.')
            plt.plot(x_efit, y_efit, 'y*')
            plt.plot(keV, energySpectrum, 'g--')
            plt.legend(loc='upper right')
            plt.title('FWHM:' + str(np.round(energyResolution*100, 2)) + '%')
            #plt.xlim(450, 600)
            plt.xlabel('Energy (keV)')
            plt.ylabel('Count (kcnts)')
            plt.savefig(filename+'_test.png')
            plt.clf()   
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)
        except:
            print("global test plot failed")
    return energyResolution_result, photoPeakBoundary, energySpectrumCount_filled, params, photoPeaksIndex

def GetPhotopeaksparams(energySpectrum, energyWindow = [450,600], params = [0,0]):
    photoPeakBoundary = [0, 0]
    invalid_number = 0
    keV, ADC, photoPeaksIndex, params, invalid = GetLinearizedEnergySpectrum(energySpectrum, params)
    
    if invalid:
        invalid_number = 3
    elif np.int64(keV[photoPeaksIndex[0]]) not in range(400, 600) and (photoPeaksIndex != [0,0]):
        invalid_number = 4
        invalid = True
        
    if not invalid:
        lowbound = energyWindow[0]
        highbound = energyWindow[1]
        lowbound_ADC = int(ADC[np.round(lowbound).astype(np.int64)])
        highbound_ADC = int(ADC[np.round(highbound).astype(np.int64)])
        photoPeakBoundary = [lowbound_ADC, highbound_ADC]
    return photoPeakBoundary, params, invalid, invalid_number, keV, ADC, photoPeaksIndex

# Get TDC bin width, assuming the width is proportional to the number of events
# in the bin
def GetTDCBinWidth(_fineTimePerCrystal):
    TDCbinWidthPerCrystal = np.zeros((numOfCrystals, numOfTDCbins), dtype=np.float64)
    for crystal in _fineTimePerCrystal:
        TDCbinWidthPerCrystal[crystal, :] = np.cumsum(_fineTimePerCrystal[crystal])/np.sum(_fineTimePerCrystal[crystal])
    return TDCbinWidthPerCrystal

def GetLookUpTableForCoarse():
    numOfBitsInCoarseCounter = 15
    LookUpTable = np.zeros(1 << 15, dtype='int16')
    LookUpTable[0x7FFF] = -1
    lfsr = np.uint32(0x0000)
    for n in range(0, (1 << numOfBitsInCoarseCounter) - 1):
        LookUpTable[lfsr] = n
        bits13_14 = lfsr >> 13
        if (bits13_14 == 0) or (bits13_14 == 3):
            newBit = np.uint8(1)
        elif (bits13_14 == 1) or (bits13_14 == 2):
            newBit = np.uint8(0)
        lfsr = (lfsr << 1) | newBit
        lfsr = lfsr & 0x7FFF

    #with open('lut.txt', 'w') as file:
    #    for n in range(0, (1 << numOfBitsInCoarseCounter)):
    #        file.write('{0:0{1}X}'.format(LookUpTable[n], 4) + ',\n')
    return LookUpTable


def correctCoarseTime(_RawCoarseTime, _FineTime, _LookUpTable, _coarseSelBit):
    realCoarseTime = _LookUpTable[_RawCoarseTime]
    # correct for coarse counter selection bit
    if _FineTime < _coarseSelBit:
        realCoarseTime += 1
    return realCoarseTime

def GetTimeResolutionforpair(timingdifference, crystal1, crystal2, filename, peak = "", plot = False, Global = False, font_size = LARGE_SIZE, clean = False, Delay = False, globalonly = False):
    x = np.linspace(-timeSpectrumWidth * coarseTimeInPicoSec,timeSpectrumWidth * coarseTimeInPicoSec, numOfBinsInTimeSpectrum)
    xnew = np.linspace(-timeSpectrumWidth * coarseTimeInPicoSec,timeSpectrumWidth * coarseTimeInPicoSec, numOfBinsInTimeSpectrum*50)
    count, binEdges = np.histogram(timingdifference,bins=numOfBinsInTimeSpectrum, range=(-timeSpectrumWidth * coarseTimeInPicoSec, timeSpectrumWidth * coarseTimeInPicoSec))
    # find the peak of the timing spectrum
    peak_tmp = np.mean(timingdifference)
    peaksIndices = peakutils.indexes(count, thres=0.3, min_dist=2)
    countOfPeaks = count[peaksIndices]
    peakIndexWithMaxCount = np.argmax(countOfPeaks)
    peakIndex = peaksIndices[peakIndexWithMaxCount]

    # confine the range of time spectrum for fitting
    expectedTimeResolution = 250  # in ps
    fitFrom = peakIndex - int(expectedTimeResolution / 50)
    fitTo = min(numOfBinsInTimeSpectrum-1, peakIndex + int(expectedTimeResolution / 50))
    timeSpectrumToFit = count[fitFrom:fitTo]
    #print(crystal1, crystal2)
    #plt.plot(count,label = 'count')
    #plt.plot(timeSpectrumToFit,label = 'timeSpectrumToFit')
    #plt.legend()
    #plt.title(str(crystal1)+'_'+str(crystal2)+'_'+str(timeSpectrumToFit.size))
    #plt.show()
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.figsize"] = (10,8)
    if timeSpectrumToFit.size < 10:
        print(crystal1,crystal2,timeSpectrumToFit.size)
        print("not enough counts for time plotting")
        return 0, peak_tmp
    # fit the time spectrum
    
    
    # print(crystal1, crystal2, Global, peak)
    # plt.rcParams.update({'font.size': font_size})
    # plt.rcParams["figure.figsize"] = (10,8)
    # plt.plot(x, count/1000, 'b-',label='Experiment')
    # title = 'crystal1: ' + str(crystal1) + '   crystal2: ' + str(crystal2) + '   FWHM: ' + str(np.round(0,2)) + '   Peak: ' + str(np.round(0,2))
    # plt.title(title)
    # plt.legend(loc='upper right')
    # plt.xlabel('Time (ps)')
    # plt.ylabel('Counts (kcnts)')
    # plt.xlim(-10000,10000)
    # plt.show()
    if Delay:
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.plot(x, count/1000, 'k-.',label='Random')
        plt.title('Delay')
        plt.legend(loc='upper right')
        plt.xlabel('Time (ps)')
        plt.ylabel('Counts (kcnts)')
        plt.xlim(-10000,10000)
        plt.ylim(0,int(np.max(count)/200))
        plt.savefig(filename)
        if not clean:
            print("Count rate with time (+- 600ps, ~ 4 FWHM) and energy windows:",np.round(np.sum(timeSpectrumToFit)/1000/60,2),"kps")
        return 0, peak_tmp
    try:
        parametersGaussian = peakutils.gaussian_fit(x[list(range(fitFrom, fitTo))], timeSpectrumToFit, center_only=False)
    except RuntimeError:
        print("DEBUG3: Fit error： Crystal1",crystal1, "Crystal2", crystal2, timingdifference)
        print("DEBUG3: ",count[fitFrom:fitTo])
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.plot(x, count/1000, 'b-',label='Experiment')
        title = 'crystal1: ' + str(crystal1) + '   crystal2: ' + str(crystal2) + '   FWHM: ' + str(np.round(0,2)) + '   Peak: ' + str(np.round(0,2))
        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlabel('Time (ps)')
        plt.ylabel('Counts (kcnts)')
        plt.xlim(-10000,10000)
        plt.savefig(filename)
        return 0, peak_tmp
    timeResolution = abs(2.355 * parametersGaussian[2])
    if peak == "":
        peak = parametersGaussian[1]
    # store the time resolution to the dictionary
    # count2, binEdges = np.histogram(timingdifference,bins=32*2*5, range=(-5 *coarseTimeInPicoSec, 5 * coarseTimeInPicoSec))
    # t = np.argmax(count2)
    # if count2[t - 32] > 0.2*count2[t]:
        # plt.plot(count2)
        # plt.show()

    if Global:
        fitCurve = peakutils.gaussian(x, parametersGaussian[0], peak, parametersGaussian[2])
        fitCurve2 = peakutils.gaussian(xnew, parametersGaussian[0], peak, parametersGaussian[2])
        r2 = r2_score(count, fitCurve)
        
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.plot(x, count/1000, 'b-',label='Experiment')
        plt.plot(xnew, fitCurve2/1000, 'r--',label=r'Fit (R^2 = '+str(np.round(r2,3))+')')
        title = 'FWHM: ' + str(np.round(timeResolution,1)) + 'ps'
        title = 'FWHM: ' + str(np.round(timeResolution,1)) + 'ps   Peak: ' + str(np.round(peak,1))+'ps'
        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlabel('Time (ps)')
        plt.ylabel('Counts (kcnts)')
        if not globalonly:
            plt.xlim(-1000,1000)
        else:
            plt.xlim(-10000,10000)
        plt.savefig(filename)
        if not clean:
            print("Count rate with time (+- 600ps, ~ 4 FWHM) and energy windows:",np.round(np.sum(timeSpectrumToFit)/1000/60,2),"kps")
    else:
        if os.path.isfile(filename.replace('.png','_bad.png')):
            os.remove(filename.replace('.png','_bad.png'))
        if plot and not os.path.isfile(filename):
            fitCurve = peakutils.gaussian(xnew, parametersGaussian[0], peak, parametersGaussian[2])
            
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)
            plt.plot(x, count, 'b-')
            plt.plot(xnew, fitCurve, 'r--')
            title = 'crystal1: ' + str(crystal1) + '   crystal2: ' + str(crystal2) + '   FWHM: ' + str(np.round(timeResolution,2)) + '   Peak: ' + str(np.round(peak,2))
            plt.title(title)
            plt.xlabel('Time (ps)')
            plt.ylabel('Counts')
            plt.savefig(filename)
            plt.clf()
            plt.rcParams.update({'font.size': font_size})
            plt.rcParams["figure.figsize"] = (10,8)

    return timeResolution, peak


def Plothist(x, filename, plot = False, time = True, font_size = LARGE_SIZE):
    if plot is True:
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
        plt.hist(x, bins = 10)
        title = 'Histogram: ' + str(np.round(np.mean(x),2))
        plt.title(title)
        if time:
            plt.xlabel('Time (ps)')
            plt.xlim(150,400)
        else:
            plt.xlabel('Emergy (%)')
            plt.xlim(0.1,0.2)
        plt.ylabel('Counts')
        plt.savefig(filename)
        plt.clf()
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams["figure.figsize"] = (10,8)
    return


def GetTimeResolution(photoPeakBoundaryPerCrystal, _coincidencePair,
        _minCountInLOR=0, energyGate=True, plot=False, dir = 0, skewlut = {}, globalonly = False, globaloffset = 0, font_size = LARGE_SIZE, clean = False, Delay = False, coincidenceTimeWindowOffset = 10):
    all_fit_time_energy_window_coutrate = 0
    start = time.time()
    original_size = _coincidencePair.shape[1]
    if energyGate is True:
        valid1 = _coincidencePair[6, :] > photoPeakBoundaryPerCrystal[_coincidencePair[4, :], 0]
        valid2 = _coincidencePair[6, :] < photoPeakBoundaryPerCrystal[_coincidencePair[4, :], 1]
        valid3 = _coincidencePair[7, :] > photoPeakBoundaryPerCrystal[_coincidencePair[5, :], 0]
        valid4 = _coincidencePair[7, :] < photoPeakBoundaryPerCrystal[_coincidencePair[5, :], 1]
        coincidencePairInPhotoPeak = _coincidencePair[:, valid1 & valid2 & valid3 & valid4]
    else: # no energy gating, not used normally
        coincidencePairInPhotoPeak = _coincidencePair
        print('No energy gating')
    coinciSum = coincidencePairInPhotoPeak.shape[1]
    print("DEBUG5: ","After energy-gating:", coinciSum, "Before energy-gating:", original_size)
    
    if not os.path.exists(dir):
        os.makedirs(dir) 
        
    #np.set_printoptions(threshold=sys.maxsize)
    if coinciSum == 0:
        return [{}, {}, [], []], [0, 0], 0
    filename = ''.join(dir.split('/')[:-3]) + '/tdc.pickle'
    print(filename)
    if not Delay:
        plt.figure()
        print('create a new figure')
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["figure.figsize"] = (10,8)
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            [t1, timePerCrystal, t2] = pickle.load(file)
        crystalID = np.concatenate([coincidencePairInPhotoPeak[4, ], coincidencePairInPhotoPeak[5, ]])
        combined_time = np.concatenate([coincidencePairInPhotoPeak[2, ], coincidencePairInPhotoPeak[3, ]])
        combined_time2 = np.float64(np.concatenate([coincidencePairInPhotoPeak[2, ], coincidencePairInPhotoPeak[3, ]]))
        crystalid_argsort = np.argsort(crystalID)
        crystalid_sorted = crystalID[crystalid_argsort]
        crystalid_split_pos = np.where(np.diff(crystalid_sorted))[0] + 1
        crystalid_unique = crystalid_sorted[np.insert(crystalid_split_pos, 0, 0)]

        index_sorted = crystalID[crystalid_argsort]
        index_split = np.split(index_sorted, crystalid_split_pos)
        print(crystalid_unique)
        for i in range(crystalid_unique.size):
            crystalid = crystalid_unique[i]
            timePerCrystal[crystalid] = np.cumsum(timePerCrystal[crystalid])/np.sum(timePerCrystal[crystalid])*32
            combined_time2[crystalID == crystalid] = timePerCrystal[crystalid][combined_time[crystalID == crystalid]]
        Fine = (np.float64(combined_time2[:coinciSum]) - np.float64(combined_time2[coinciSum:]))* coarseTimeInPicoSec/32
        print(np.float64(combined_time2[:coinciSum]),np.float64(combined_time2[coinciSum:]))
    Fine = (np.float64(coincidencePairInPhotoPeak[8, ]) - np.float64(coincidencePairInPhotoPeak[9, ]))* coarseTimeInPicoSec/TDCbinwidth
    #Fine = (np.float64(coincidencePairInPhotoPeak[2, ]) - np.float64(coincidencePairInPhotoPeak[3, ]))* coarseTimeInPicoSec/32
    Coarse = (np.float64(coincidencePairInPhotoPeak[0, ]) - np.float64(coincidencePairInPhotoPeak[1, ]))*coarseTimeInPicoSec
    timeDifference = np.zeros((3, coinciSum), dtype=np.int32) 
    timeDifference[0, :] = coincidencePairInPhotoPeak[4, ]#cyrstal 1
    timeDifference[1, :] = coincidencePairInPhotoPeak[5, ]#crystal 2
    if((np.float64(coincidencePairInPhotoPeak[8, ]) == np.float64(coincidencePairInPhotoPeak[0, ])).all()): # coincidence filtered compact mode
        timeDifference[2, :] = np.float64(coincidencePairInPhotoPeak[0, ])*coarseTimeInPicoSec/TDCbinwidth - globaloffset
    else:
        timeDifference[2, :] = Coarse + Fine - globaloffset
    #index = (coincidencePairInPhotoPeak[4, ] == 0)
    #print(coincidencePairInPhotoPeak[:,index])
    #value = timeDifference[2, index]
    #plt.hist(value, np.linspace(min(value),max(value),int((max(value)-min(value) + 1)/50)))
    
    #value = timeDifference[1, index]
    #plt.hist(value, np.linspace(min(value)-0.5,max(value)+0.5,int((max(value)-min(value) + 2))))
    #plt.show()
    #sdfsd
    # sort coincidence according to coarse value 
    timeDifference_split_global = []
    timeResolutionPerLOR = {}
    timeDifference_split = []
    timeDifference_unique_crystal1 = []
    timeDifference_unique_crystal2 = [] 
    if globalonly:
        timeDifference_global = timeDifference[2, ]
        all_fit_time_energy_window_coutrate = timeDifference.shape[1]
        if not isinstance(skewlut, dict):
            timeDifference_global = timeDifference_global - skewlut[timeDifference[0, ],timeDifference[1, ]]
    elif not Delay:
        timeDifference_argsort = np.lexsort((timeDifference[1, ], timeDifference[0, ]))
        timeDifference_sorted = timeDifference[:, timeDifference_argsort]
        timeDifference_split_pos = np.where(np.diff(timeDifference_sorted[1, ]))[0] + 1
        timeDifference_unique_crystal1 = timeDifference_sorted[0, np.insert(timeDifference_split_pos, 0, 0)]
        timeDifference_unique_crystal2 = timeDifference_sorted[1, np.insert(timeDifference_split_pos, 0, 0)]
        timeDifference_split = np.split(timeDifference_sorted[2, ], timeDifference_split_pos)
        # Get the timing spetrum and timing resolution per LOR
        i = 0
        
        #t = timeDifference_sorted[2, ] - offset
        #z = np.histogram(t)
        #print(z[1][np.argmax(z[0])])
        #plt.hist(timeDifference_sorted[2, ]-z[1][np.argmax(z[0])])
        #plt.xlim(-1e4,1e4)
        #plt.show()
        #print(timeDifference_split)
        print('Maximum counts in LOR:', np.max(np.array([i.size for i in timeDifference_split])))
        print('LOR number:', len(timeDifference_split))
        while i < len(timeDifference_split):
            # combining all LORs to plot global only
            peak = ""
            # get skew from skew file
            crystal1 = timeDifference_unique_crystal1[i]
            crystal2 = timeDifference_unique_crystal2[i]
            #print(crystal1,crystal2)
            
            if not isinstance(skewlut, dict):
                peak = skewlut[crystal1, crystal2]
            if globalonly:
                if peak == "":
                    peak = 0
            else:
                all_fit_time_energy_window_coutrate += timeDifference_split[i].size
                
                #print(crystal1,crystal2)
                #print(timeDifference_split[i])
                
                #print(timeDifference_split[i].size,crystal1, crystal2)
                if (timeDifference_split[i].size < _minCountInLOR): #| (isinstance(skewlut,dict) & (peak == "")):
                    timeResolution = 0
                    if(peak == ""):
                        peak = np.mean(timeDifference_split[i])
                #if timeDifference_split[i].size < 0:
                    #del timeDifference_split[i]
                    #timeDifference_unique_crystal1 = np.delete(timeDifference_unique_crystal1, i)
                    #timeDifference_unique_crystal2 = np.delete(timeDifference_unique_crystal2, i)
                    #continue
                #print("crystal1,crystal2,counts): ",crystal1,crystal2,len(timeDifference_split[i]))
                #print(min(timeDifference_split[i]),max(timeDifference_split[i]))
                #plt.hist(timeDifference_split[i],np.linspace(min(timeDifference_split[i]),max(timeDifference_split[i]),max(timeDifference_split[i]) - min(timeDifference_split[i]) + 1))
                #plt.show()
                else:
                    filename = dir + '/' + str(crystal1)+'_' + str(crystal2) + '.png'
                    #timeResolution = 1
                    #peak = 0
                    timeResolution, peak = GetTimeResolutionforpair(timeDifference_split[i], crystal1, crystal2, filename, peak = peak,  plot = plot, Global = False)
                '''if timeResolution == 0:
                    del timeDifference_split[i]
                    timeDifference_unique_crystal1 = np.delete(timeDifference_unique_crystal1, i)
                    timeDifference_unique_crystal2 = np.delete(timeDifference_unique_crystal2, i)
                    continue'''
                # store the time resolution to the dictionary
                if crystal1 in timeResolutionPerLOR:
                    timeResolutionPerLOR[crystal1][crystal2] = np.array([timeResolution, peak,timeDifference_split[i].size])
                else:
                    timeResolutionPerLOR[crystal1] = {crystal2: np.array([timeResolution, peak,timeDifference_split[i].size])}

            if timeResolution != 0:
                timeDifference_split_global.append(timeDifference_split[i] - peak)
            i += 1
        if len(timeDifference_split_global) != 0:
            timeDifference_global = np.concatenate(timeDifference_split_global)
        else:
            print(len(timeDifference_sorted[2, ]))
            timeDifference_global = timeDifference_sorted[2, ]
        
    if Delay:
        filename = os.path.split(dir)[0] + '/Delay_timing.png'
        timeDifference_global = timeDifference_global - coincidenceTimeWindowOffset*coarseTimeInPicoSec 
    else:    
        if not clean:
            print("Count rate with time and energy windows:",np.round(all_fit_time_energy_window_coutrate/1000/60,2),"kps")
        filename = os.path.split(dir)[0] + '/Global_timing.png'
    timeResolution, peak = GetTimeResolutionforpair(timeDifference_global, 0, 0, filename, peak = "", plot = True, Global = True, Delay = Delay, globalonly = globalonly)
    
    return [timeResolutionPerLOR, timeDifference_split, timeDifference_unique_crystal1, timeDifference_unique_crystal2], [timeDifference_global.size, timeResolution], coinciSum

def calibrate(timeDifference_split, timeDifference_unique_crystal1, timeDifference_unique_crystal2, timeResolutionPerLOR, crystal_dict, argsort = {}, second = False):
    if(second):
        crystal = timeDifference_unique_crystal2[argsort[0]]
    else:
        crystal = timeDifference_unique_crystal1[0]
    sum = 0
    num = 0
    crystal_tmp = 0
    for i in range(len(timeDifference_split)):
        if second:
            index = argsort[i]
            crystal1 = timeDifference_unique_crystal1[index]
            crystal2 = timeDifference_unique_crystal2[index]
            crystal_tmp = crystal2
        else:
            crystal1 = timeDifference_unique_crystal1[i]
            crystal2 = timeDifference_unique_crystal2[i]
            crystal_tmp = crystal1
        if crystal1 not in crystal_dict.keys():
            crystal_dict[crystal1] = 0
        if crystal2 not in crystal_dict.keys():
            crystal_dict[crystal2] = 0
        array = timeResolutionPerLOR[crystal1][crystal2]
        peak = array[1] + crystal_dict[crystal2] - crystal_dict[crystal1]
        if second:
            peak = -peak
        if crystal_tmp not in crystal_dict.keys():
            crystal_dict[crystal_tmp] = 0
        if(crystal_tmp == crystal):
            sum += peak
            num += 1
        else:
            crystal_dict[crystal] += sum / num
            crystal = crystal_tmp
            sum = peak
            num = 1
    crystal_dict[crystal_tmp] += sum / num

    return crystal_dict
    

def savelistmode(coincidencePair, listmodeFileDirectory, fdir, skewlut, randomize = True): 
    if not os.path.exists(listmodeFileDirectory):
        os.makedirs(listmodeFileDirectory)
    with open('geometry.pickle', 'rb') as file:
        crystalPositionMap = pickle.load(file)
    if randomize:
       rawData = np.transpose(coincidencePair)
       np.random.shuffle(rawData)
       coincidencePair = np.transpose(rawData)
    listmodeFile = listmodeFileDirectory + 'resPhantomRotateClockwise.lm'
    textOutputFile = listmodeFileDirectory + 'resPhantomRotateClockwise.txt'
    # coincidencePairData = np.loadtxt(dataFile, max_rows=7000000)
    numberOfCoincidenceEvents = coincidencePair.shape[1]

    val = 0
    offset = np.zeros((16*6*8*18,16*6*8*18))
    for crystal1 in range(16*6*8*18):
        for crystal2 in range(16*6*8*18):
            if crystal1 in skewlut.keys():
                if crystal2 in skewlut[crystal1].keys():
                    offset[crystal1,crystal2] = skewlut[crystal1][crystal2][1]
                    offset[crystal2,crystal1] = -skewlut[crystal1][crystal2][1]
            if crystal2 in skewlut.keys():
                if crystal1 in skewlut[crystal2].keys():
                    offset[crystal2,crystal1] = skewlut[crystal2][crystal1][1]
                    offset[crystal1,crystal2] = -skewlut[crystal2][crystal1][1]

    crystalID1_arr = np.uint16(coincidencePair[4,:])
    tmp = crystalID1_arr
    firstCrystalX_arr = crystalPositionMap[tmp,0]
    crystalID2_arr = np.uint16(coincidencePair[5,:])
    tmp = crystalID2_arr
    SecondCrystalX_arr = crystalPositionMap[tmp,0]
    invalid = ((SecondCrystalX_arr == 0)|(firstCrystalX_arr == 0))
    coincidencePair = coincidencePair[:,~invalid]
    crystalID1_arr = np.uint16(coincidencePair[4,:])
    tmp = crystalID1_arr
    firstCrystalX_arr = crystalPositionMap[tmp,0]
    firstCrystalY_arr = crystalPositionMap[tmp,1]
    firstCrystalZ_arr = crystalPositionMap[tmp,2]
    crystalID2_arr = np.uint16(coincidencePair[5,:])
    tmp = crystalID2_arr
    SecondCrystalX_arr = crystalPositionMap[tmp,0]
    SecondCrystalY_arr = crystalPositionMap[tmp,1]
    SecondCrystalZ_arr = crystalPositionMap[tmp,2]
    diff = np.float64(coincidencePair[0,:] - coincidencePair[1,:])*1600  +  np.float64(coincidencePair[8,:] - coincidencePair[9,:])*1600/1024
    TOF_arr = speedOfLight * np.float64((diff - offset[crystalID1_arr,crystalID2_arr]) * math.pow(10, -12))
    numberOfCoincidenceEvents = coincidencePair.shape[1]
    percent_bar = 0.05
    with open(listmodeFile, 'wb') as lm:        
        with open(textOutputFile, 'a') as f:
            print(numberOfCoincidenceEvents)
            for i in range(numberOfCoincidenceEvents):
                if(i > percent_bar*numberOfCoincidenceEvents):
                    print("Finish ",int(i*100/numberOfCoincidenceEvents),'%')
                    percent_bar += 0.05
                # print(line)
                crystalID1 = crystalID1_arr[i]
                firstCrystalX = firstCrystalX_arr[i]
                firstCrystalY = firstCrystalY_arr[i]
                firstCrystalZ = firstCrystalZ_arr[i]
                crystalID2 = crystalID2_arr[i]
                SecondCrystalX = SecondCrystalX_arr[i]
                SecondCrystalY = SecondCrystalY_arr[i]
                SecondCrystalZ = SecondCrystalZ_arr[i]
                TOF = TOF_arr[i]
                
                eventPair = np.zeros((1, 10))
                eventPair[0, :] = np.array([firstCrystalX, firstCrystalY, firstCrystalZ, TOF, val, SecondCrystalX, SecondCrystalY, SecondCrystalZ, crystalID1, crystalID2])
                # print(eventPair)
                textOutputFmt = ['%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f', '%5.2f']
                if(i < 1000):
                    np.savetxt(f, eventPair, fmt=textOutputFmt)
                # eventToWrite = array('f', [firstCrystalX, firstCrystalY, firstCrystalZ, TOF, val, SecondCrystalX, SecondCrystalY, SecondCrystalZ, val, val])
                # eventToWrite.tofile(lm)

                lm.write(struct.pack('f', float(firstCrystalX)))
                lm.write(struct.pack('f', float(firstCrystalY)))
                lm.write(struct.pack('f', float(firstCrystalZ)))
                lm.write(struct.pack('f', float(TOF)))
                lm.write(struct.pack('f', float(val)))
                lm.write(struct.pack('f', float(SecondCrystalX)))
                lm.write(struct.pack('f', float(SecondCrystalY)))
                lm.write(struct.pack('f', float(SecondCrystalZ)))
                lm.write(struct.pack('f', float(val)))
                lm.write(struct.pack('f', float(val)))
