numOfSubmodulesPerModule = 6
numOfHalfchipPerSubmodule = 8
numOfChannelPerHalfchip = 18


import numpy as np
import pickle
class Singles:
    def __init__(self, eventBytes,compact = True, keephighestenergy = False, delay = False, savetxt = False):
    
        self.frame = self.GetTimeFrame(eventBytes,compact)
        self.channelid = self.GetChannelID(eventBytes)
        self.submoduleid = self.GetSubmoduleID(eventBytes)
        self.moduleid = self.GetModuleID(eventBytes,compact)
        self.valid = self.IsValid(eventBytes)
        eventBytes = eventBytes[self.valid, ...]
        self.frame = self.frame[self.valid,...]
        self.keep = self.valid[self.valid, ...]
        self.size = self.keep.size
        self.valid = self.valid[self.valid, ...]
        self.coarse = self.GetCoarseTime(eventBytes)
        self.moduleid = self.GetModuleID(eventBytes,compact)
        self.subframe = self.Getsubframe(eventBytes,compact)
        self.submoduleid = self.GetSubmoduleID(eventBytes)
        self.halfchipid = self.GetHalfChipID(eventBytes)
        self.channelid = self.GetChannelID(eventBytes)
        self.energy = self.GetEnergy(eventBytes)
        self.crystalID = self.GetCrystalId()
        self.fine = self.GetFineTime(eventBytes)
        self.moduleid = self.GetModuleID(eventBytes,compact)
        self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
        if keephighestenergy:
            print(self.coarse[:20])
            print(self.energy[:20])
            argsort = np.lexsort((self.energy*(-1), self.coarse, self.frame))
            eventBytes = eventBytes[argsort, ...]
            self.frame = self.frame[argsort,...]
            self.coarse = self.GetCoarseTime(eventBytes)
            
            
            print(self.coarse[:20])
            print(self.energy[:20])
            
            valid = (np.diff(self.coarse) > 10)
            valid = np.insert(valid, 0, True)
            eventBytes = eventBytes[valid, ...]
            self.frame = self.frame[valid,...]
            
            self.channelid = self.GetChannelID(eventBytes)
            self.submoduleid = self.GetSubmoduleID(eventBytes)
            self.moduleid = self.GetModuleID(eventBytes,compact)
            self.valid = self.IsValid(eventBytes)
            eventBytes = eventBytes[self.valid, ...]
            self.keep = self.valid[self.valid, ...]
            self.size = self.keep.size
            self.frame = self.frame[self.valid,...]
            self.moduleid = self.GetModuleID(eventBytes,compact)
            self.subframe = self.Getsubframe(eventBytes,compact)
            self.submoduleid = self.GetSubmoduleID(eventBytes)
            self.halfchipid = self.GetHalfChipID(eventBytes)
            self.channelid = self.GetChannelID(eventBytes)
            self.energy = self.GetEnergy(eventBytes)
            self.crystalID = self.GetCrystalId()
            self.fine = self.GetFineTime(eventBytes)
            self.fine_ps = self.GetFinepsTime(eventBytes)
            self.coarse = self.GetCoarseTime(eventBytes)
            self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
            #print(self.frame)
        self.moduleid = self.GetModuleID(eventBytes,compact)
        self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
        if savetxt:
            with open('data.txt', 'w') as f:
                for i in range(int(self.index.shape[0]/2)):
                    f.write("{:08x}".format(self.coarse[i*2+1]) +" "+  "{:08x}".format(self.coarse[i*2]) +" "+"{:08x}".format(self.index[i*2+1]) +" "+ "{:08x}".format(self.index[i*2]))
                    f.write("\n")
            f.close()
    def IsValid(self, _eventBytes):
        hit_Master_testHit = _eventBytes[:, 2] & 0b11100000
        bad = _eventBytes[:, 4] & 0b10000000
        valid1 = (hit_Master_testHit == 192)
        valid2 = (self.moduleid < 16)
        valid3 = (self.channelid < 18)
        valid4 = (self.submoduleid < 6)
        valid5 = (bad == 128)
        #valid_tmp = (self.moduleid == 0) | (self.moduleid == 8)
        #return valid2&valid4
        return valid1 & valid2 & valid3 & valid4 & valid5
    def GetChannelID(self, _eventBytes):
        channelID = _eventBytes[:, 2] & 0b00011111
        #return _eventBytes[:, 2]
        return channelID
    def GetEnergy(self, _eventBytes):
        bit8 = np.uint16(_eventBytes[:, 5] & 0b10000000) << 1
        bit7_0 = _eventBytes[:, 7]
        #return bit7_0
        return bit8 + bit7_0
    def GetModuleID(self, _eventBytes,compact):
        if compact:
            ModuleID = _eventBytes[:, 1] & 0b00001111
        else:
            ModuleID = (_eventBytes[:, 0] & 0b11000000) >> 6
        return ModuleID
    def GetSubmoduleID(self, _eventBytes):
        submoduleID = (_eventBytes[:, 0] & 0b00111000) >> 3
        return submoduleID
    def GetHalfChipID(self, _eventBytes):
        halfChipID = _eventBytes[:, 0] & 0b00000111
        return halfChipID
    def Getsubframe(self, _eventBytes, compact):
        if not compact:
            subframe = _eventBytes[:, 1]
        else:
            subframe = (_eventBytes[:, 1] & 0b11110000) >> 4
        return subframe   
    def GetTimeFrame(self, _eventBytes, compact):
        frame = (np.uint32(_eventBytes[:, 4]) << 8*0) + (np.uint32(_eventBytes[:, 5]) << 8*1) + (np.uint32(_eventBytes[:, 6]) << 8*2) + np.uint32(_eventBytes[:, 7] << 8*3)
        frame[_eventBytes[:,0] != 0xfa] = 0
        pos = np.where(frame > 0)[0]
        prev_pos = pos[0]
        for i in pos[1:]:
            frame[prev_pos:i] = frame[prev_pos]
            prev_pos = i
        frame[pos[-1]:] = frame[pos[-1]]
        return frame
    def GetFineTime(self, _eventBytes):
        finepsTime_9_8 = np.uint16(_eventBytes[:, 4] & 0b01100000) << 3
        finepsTime_7_0 = _eventBytes[:, 3]
        return finepsTime_9_8 + finepsTime_7_0
    def GetCoarseTime(self, _eventBytes):
        coarseBit_14_8 = np.uint16(_eventBytes[:, 5] & 0b01111111) << 8
        coarseBit_7_0 = _eventBytes[:, 6]
        return coarseBit_14_8 + coarseBit_7_0
    def GetCrystalId(self):
        # get (flatten) crystal ID from (module ID * 6 * 8 * 18 + submoduleID *
        # 8 * 18 + HalfChipID * 18 + channelID):
        return np.uint16(self.moduleid) * 864 + np.uint16(self.submoduleid) * 144 + np.uint16(self.halfchipid) * 18 + self.channelid
    def GetKeepEvents(self, keep_index):
        self.size = len(keep_index)
        self.keep = self.keep[keep_index]
        self.energy = self.energy[keep_index]
        self.moduleid = self.moduleid[keep_index]
        self.submoduleid = self.submoduleid[keep_index]
        self.halfchipid = self.halfchipid[keep_index]
        self.channelid = self.channelid[keep_index]
        self.crystalID = self.crystalID[keep_index]
        self.frame = self.frame[keep_index]
        self.fine = self.fine[keep_index]
        self.coarse = self.coarse[keep_index]
        self.index = keep_index
        return self
    def AddKeepEvents(self, event_keep):
        if event_keep == []:
            return
        self.size = self.size + event_keep.size
        self.keep = np.concatenate((self.keep, event_keep.keep))
        self.energy = np.concatenate((self.energy, event_keep.energy))
        self.moduleid = np.concatenate((self.moduleid, event_keep.moduleid))
        self.submoduleid = np.concatenate((self.submoduleid, event_keep.submoduleid))
        self.halfchipid = np.concatenate((self.halfchipid, event_keep.halfchipid))
        self.channelid = np.concatenate((self.channelid, event_keep.channelid))
        self.crystalID = np.concatenate((self.crystalID, event_keep.crystalID))
        self.frame = np.concatenate((self.frame, event_keep.frame))
        self.fine = np.concatenate((self.fine, event_keep.fine))
        self.coarse = np.concatenate((self.coarse, event_keep.coarse))
        self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
        return self



class Coincidences:
    def __init__(self, eventBytes):
        self.frame = self.GetTimeFrame(eventBytes)
        self.valid = self.IsValid(eventBytes)
        self.frame = self.frame[self.valid]
        self.frame1 = self.frame
        eventBytes = eventBytes[self.valid]
        self.valid = self.IsValid(eventBytes)
        self.channelid, self.channelid1 = self.GetChannelID(eventBytes)
        self.moduleid, self.moduleid1 = self.GetModuleID(eventBytes)
        self.submoduleid, self.submoduleid1 = self.GetSubmoduleID(eventBytes)
        self.halfchipid, self.halfchipid1 = self.GetHalfChipID(eventBytes)
        self.channelid, self.channelid1 = self.GetChannelID(eventBytes)
        self.energy, self.energy1 = self.GetEnergy(eventBytes)
        self.fine_ps,self.fine_ps1 = self.GetFinepsTime(eventBytes)
        self.coarse, self.coarse1 = self.GetCoarseTime(eventBytes)
        self.crystalID, self.crystalID1 = self.GetCrystalId()
        self.Swap(self.crystalID > self.crystalID1)
        self.size = self.valid.size
        self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
    def IsValid(self, _eventBytes):
        valid = (_eventBytes[:, 0] & 0xC0 == 0)
        return valid
        
    def Swap(self,index):
        tmp = self.channelid[index]
        self.channelid[index] = self.channelid1[index]
        self.channelid1[index] = tmp
        tmp = self.moduleid[index]
        self.moduleid[index] = self.moduleid1[index]
        self.moduleid1[index] = tmp
        tmp = self.submoduleid[index]
        self.submoduleid[index] = self.submoduleid1[index]
        self.submoduleid1[index] = tmp
        tmp = self.halfchipid[index]
        self.halfchipid[index] = self.halfchipid1[index]
        self.halfchipid1[index] = tmp
        tmp = self.crystalID[index]
        self.crystalID[index] = self.crystalID1[index]
        self.crystalID1[index] = tmp
        tmp = self.energy[index]
        self.energy[index] = self.energy1[index]
        self.energy1[index] = tmp
        tmp = self.fine_ps[index]
        self.fine_ps[index] = self.fine_ps1[index]
        self.fine_ps1[index] = tmp
        return  
    def GetTimeFrame(self, _eventBytes):
        frame = (np.uint32(_eventBytes[:, 4]) << 8*0) + (np.uint32(_eventBytes[:, 5]) << 8*1) + (np.uint32(_eventBytes[:, 6]) << 8*2) + np.uint32(_eventBytes[:, 7] << 8*3)
        frame[_eventBytes[:,0] != 0xba] = 0
        pos = np.where(frame > 0)[0]
        prev_pos = pos[0]
        for i in pos[1:]:
            frame[prev_pos:i] = frame[prev_pos]
            prev_pos = i
        frame[pos[-1]:] = frame[pos[-1]]
        return frame   
    def GetChannelID(self, _eventBytes):
        channelID1 = ((_eventBytes[:, 5] & 0x0F) << 1) + ((_eventBytes[:, 6] & 0x80) >> 7 )
        channelID2 = _eventBytes[ :, 2 ] & 0x1F ;
        return channelID1, channelID2

    def GetEnergy(self, _eventBytes):
        energy1 = (np.uint16(_eventBytes[:,  2 ]       ) << 1 ) + ((_eventBytes[:,  3 ] & 0x80) >> 7)
        energy2 = (np.uint16(_eventBytes[ :, 3 ] & 0x7F) << 2 ) + (np.uint16(_eventBytes[ :, 4 ] & 0xC0) >> 6)
        return energy1, energy2

    def GetModuleID(self, _eventBytes):
        ModuleID1 = (_eventBytes[ :, 4 ] & 0x3C) >>2 
        ModuleID2 = (_eventBytes[ :, 6 ] & 0x78) >>3 
        return ModuleID1, ModuleID2

    def GetSubmoduleID(self, _eventBytes):
        submoduleID1 = ((_eventBytes[ :, 4 ] & 0x03) << 1) + ((_eventBytes[ :, 5 ] & 0x80) >> 7)
        submoduleID2 = _eventBytes[:,  6 ] & 0x07
        return submoduleID1, submoduleID2

    def GetHalfChipID(self, _eventBytes):
        halfChipID1 = (_eventBytes[ :, 5 ] & 0x70) >> 4
        halfChipID2 = (_eventBytes[ :, 7 ] & 0xE0) >> 5
        return halfChipID1, halfChipID2

    def GetFinepsTime(self, _eventBytes):
        finepsTime1 = (np.uint16(_eventBytes[:, 0 ]& 0x03) <<8 )  + _eventBytes[ :, 1 ]
        finepsTime2 = finepsTime1*0
        return finepsTime1, finepsTime2

    def GetCoarseTime(self, _eventBytes):
        coarse1 = (np.uint16(_eventBytes[:, 0 ]& 0x3C) >> 2 )
        coarse2 = coarse1*0
        return coarse1, coarse2

    def GetCrystalId(self):
        # get (flatten) crystal ID from (module ID * 6 * 8 * 18 + submoduleID *
        # 8 * 18 + HalfChipID * 18 + channelID):
        crystalid0 = np.uint16(self.moduleid) * 864 + np.uint16(self.submoduleid) * 144 + np.uint16(self.halfchipid) * 18 + self.channelid
        crystalid1 = np.uint16(self.moduleid1) * 864 + np.uint16(self.submoduleid1) * 144 + np.uint16(self.halfchipid1) * 18 + self.channelid1
        return crystalid0, crystalid1

class Coincidences_C:
    def __init__(self, eventBytes, delay = False):
        self.frame = self.GetTimeFrame(eventBytes)
        self.timediff = self.GetTime(eventBytes)
        self.energy, self.energy1 = self.GetEnergy(eventBytes)
        self.moduleid, self.moduleid1 = self.GetModuleID(eventBytes)
        self.submoduleid, self.submoduleid1 = self.GetSubmoduleID(eventBytes)
        self.channelid, self.channelid1 = self.GetChannelID(eventBytes)
        self.valid = self.IsValid(eventBytes, delay)
        self.frame = self.frame[self.valid]
        self.frame1 = self.frame
        eventBytes = eventBytes[self.valid]
        self.timediff = self.GetTime(eventBytes)
        self.valid = self.valid[self.valid]
        self.moduleid, self.moduleid1 = self.GetModuleID(eventBytes)
        self.submoduleid, self.submoduleid1 = self.GetSubmoduleID(eventBytes)
        self.halfchipid, self.halfchipid1 = self.GetHalfChipID(eventBytes)
        self.channelid, self.channelid1 = self.GetChannelID(eventBytes)
        self.energy, self.energy1 = self.GetEnergy(eventBytes)
        self.timediff = self.GetTime(eventBytes)
        self.crystalID, self.crystalID1 = self.GetCrystalId()
        self.Swap(self.crystalID > self.crystalID1)
        self.coarse = self.timediff
        self.coarse1 = self.timediff
        self.fine_ps = self.timediff
        self.fine_ps1 = self.timediff
        self.fine = self.timediff
        self.fine1 = self.timediff
        self.size = self.valid.size
        self.index = np.linspace(0, self.size - 1, self.size, dtype = np.int64)
        
    def IsValid(self, _eventBytes,delay):
        valid2 = (self.energy > 10) & ( self.energy1 > 10) & (self.energy < 500) & ( self.energy1 < 500)
        valid3 = (self.submoduleid < 6) & ( self.submoduleid1 < 6) 
        valid4 = (self.channelid < 18) & ( self.channelid1 < 18) 
        #valid_tmp = (self.moduleid == 0) | (self.moduleid == 8)
        #valid_tmp2 = (self.moduleid1 == 0) | (self.moduleid1 == 8)
        if(delay):
            valid = (((_eventBytes[:, 0] & 0x40) >> 6) == 1)&~((_eventBytes[:, 1]==0x55)&(_eventBytes[:, 0] & 0x0F==0x0A))
        else:
            valid = (((_eventBytes[:, 0] & 0x40) >> 6) == 0)&~((_eventBytes[:, 1]==0x55)&(_eventBytes[:, 0] & 0x0F==0x0A))
        return valid & valid2 & valid3 & valid4
        
    def Swap(self,index):
        tmp = self.channelid[index]
        self.channelid[index] = self.channelid1[index]
        self.channelid1[index] = tmp
        tmp = self.moduleid[index]
        self.moduleid[index] = self.moduleid1[index]
        self.moduleid1[index] = tmp
        tmp = self.submoduleid[index]
        self.submoduleid[index] = self.submoduleid1[index]
        self.submoduleid1[index] = tmp
        tmp = self.halfchipid[index]
        self.halfchipid[index] = self.halfchipid1[index]
        self.halfchipid1[index] = tmp
        tmp = self.crystalID[index]
        self.crystalID[index] = self.crystalID1[index]
        self.crystalID1[index] = tmp
        tmp = self.energy[index]
        self.energy[index] = self.energy1[index]
        self.energy1[index] = tmp
        self.timediff[index] = -self.timediff[index]
        return  
    def GetTimeFrame(self, _eventBytes):
        frame = (np.uint32(_eventBytes[:, 4]) << 8*0) + (np.uint32(_eventBytes[:, 5]) << 8*1) + (np.uint32(_eventBytes[:, 6]) << 8*2) + np.uint32(_eventBytes[:, 7] << 8*3)
        frame[_eventBytes[:,0] != 0xba] = 0
        pos = np.where(frame > 0)[0]
        prev_pos = pos[0]
        for i in pos[1:]:
            frame[prev_pos:i] = frame[prev_pos]
            prev_pos = i
        frame[pos[-1]:] = frame[pos[-1]]
        return frame   
        
    def GetChannelID(self, _eventBytes):
        channelID1 = ((_eventBytes[:,  5 ] & 0x0F ) << 1)  + ((_eventBytes[:,  6 ]& 0x80) >> 7)
        channelID2 = (_eventBytes[:,  7 ] & 0x1F)
        return channelID1, channelID2

    def GetEnergy(self, _eventBytes):
        energy1 = (np.uint16(_eventBytes[:,  2 ]       ) << 1)  + (np.uint16(_eventBytes[:,  3 ] & 0x80) >> 7)
        energy2 = (np.uint16(_eventBytes[ :, 3 ] & 0x7F) << 2 ) + (np.uint16(_eventBytes[ :, 4 ] & 0xC0) >> 6)
        return energy1, energy2

    def GetModuleID(self, _eventBytes):
        ModuleID1 = (_eventBytes[:,  4 ] & 0x3C) >> 2
        ModuleID2 = (_eventBytes[:,  6 ] & 0x78) >> 3
        return ModuleID1, ModuleID2

    def GetSubmoduleID(self, _eventBytes):
        submoduleID1 = ((_eventBytes[:,  4 ] & 0x03 ) << 1)  + ((_eventBytes[:,  5 ]& 0x80) >> 7)
        submoduleID2 = _eventBytes[:,  6 ] & 0x07
        return submoduleID1, submoduleID2

    def GetHalfChipID(self, _eventBytes):
        halfChipID1 = (_eventBytes[ :, 5 ] & 0x70) >> 4
        halfChipID2 = (_eventBytes[ :, 7 ] & 0xE0) >> 5
        return halfChipID1, halfChipID2
 
        
    def GetTime(self, _eventBytes):
        time = (np.int16(_eventBytes[:, 0 ]& 0x3F) <<8 )  + np.int16(_eventBytes[ :, 1 ])
        return time

    def GetCrystalId(self):
        # get (flatten) crystal ID from (module ID * 6 * 8 * 18 + submoduleID *
        # 8 * 18 + HalfChipID * 18 + channelID):
        crystalid0 = np.uint16(self.moduleid) * 864 + np.uint16(self.submoduleid) * 144 + np.uint16(self.halfchipid) * 18 + self.channelid
        crystalid1 = np.uint16(self.moduleid1) * 864 + np.uint16(self.submoduleid1) * 144 + np.uint16(self.halfchipid1) * 18 + self.channelid1
        return crystalid0, crystalid1