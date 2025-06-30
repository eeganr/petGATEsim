import numpy as np
import randoms
import pickle

CYCLE = 1.6e-9  # clock cycle (s)
TAU = 3 * CYCLE  # coincidence window (s)
DELAY = 10 * CYCLE  # delay for DW estimate (s)
DETECTORS = 48 * 48
FILE_RANGE = range(1, 100 + 1)

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
        self.time = np.array(self.coarse) * 1.6e-9 + np.array(self.fine) * 50e-12
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

def load_file(n):
    with open(f'/home/users/eeganr/petGATEsim/gen2process/singles_1.pkl', 'rb') as f:
        results = pickle.load(f)
    return results

singles = load_file(1)

