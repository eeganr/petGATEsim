# === User Configuration ===
INFILE     = "gateCoincidences.dat"  # Path to your .dat file (hits, singles, or coincidences)
FILE_TYPE  = "coinc"                  # Choose: "hits", "singles" or "coinc"
NUM_VOLIDS = 6                         # Number of volume ID levels (e.g. 6 for cylindricalPET)
# USER_MASK: None = include all specs; otherwise provide a 0/1 list
#   length = 20 for hits, 18 for singles, 36 for coincidences
#   mask = [1]*36
USER_MASK  = None
OUTCSV     = "output.csv"           # Path to output CSV

import struct
import os
import pandas as pd

# Build specs for gateHits.dat (.dat)
def build_hits_specs(num_vol_ids):
    return [
        ("run",             "i", 1),
        ("event",           "i", 1),
        ("primaryID",       "i", 1),
        ("sourceID",        "i", 1),
        ("volIDs",          "i", num_vol_ids),
        ("time",            "d", 1),
        ("Edep",            "d", 1),
        ("range",           "d", 1),
        ("posX",            "d", 1),
        ("posY",            "d", 1),
        ("posZ",            "d", 1),
        ("geant4_code",     "i", 1),
        ("particleID",      "i", 1),
        ("motherID",        "i", 1),
        ("photonID",        "i", 1),
        ("nComPhantom",     "i", 1),
        ("nRayPhantom",     "i", 1),
        ("process",         "8s",1),
        ("lastComptonVol",  "8s",1),
        ("lastRayleighVol", "8s",1),
    ]

# Build specs for gateSingles.dat
def build_singles_specs(num_vol_ids):
    return [
        ("run",       "i", 1),
        ("event",     "i", 1),
        ("srcID",     "i", 1),
        ("srcX",      "d", 1),
        ("srcY",      "d", 1),
        ("srcZ",      "d", 1),
        ("volIDs",    "i", num_vol_ids),
        ("time",      "d", 1),
        ("Edep",      "d", 1),
        ("detX",      "d", 1),
        ("detY",      "d", 1),
        ("detZ",      "d", 1),
        ("nComPh",    "i", 1),
        ("nComDet",   "i", 1),
        ("nRayPh",    "i", 1),
        ("nRayDet",   "i", 1),
        ("phantomCom","8s",1),
        ("phantomRay","8s",1),
    ]

# Build specs for gateCoincidences.dat by combining two singles
def build_coinc_specs(num_vol_ids):
    single = build_singles_specs(num_vol_ids)
    specs = []
    for i in (1, 2):
        for name, fmt, cnt in single:
            specs.append((f"{name}{i}", fmt, cnt))
    return specs

# Convert specs + mask into struct and field names
def build_struct_and_fields(specs, mask):
    if len(mask) != len(specs):
        raise ValueError(f"Mask length {len(mask)} != number of specs {len(specs)}")
    fmt = "<"
    fields = []
    for keep, (name, fchar, cnt) in zip(mask, specs):
        if keep:
            fmt += f"{cnt}{fchar}"
            if cnt == 1:
                fields.append(name)
            else:
                for idx in range(cnt):
                    fields.append(f"{name}_{idx}")
    return struct.Struct(fmt), fields

# Read .dat file and yield each record as a dict
def parse_file(path, st, fields):
    with open(path, "rb") as f:
        while True:
            chunk = f.read(st.size)
            if not chunk:
                break
            yield dict(zip(fields, st.unpack(chunk)))

# Main
if __name__ == "__main__":
    # Choose the right specs
    if FILE_TYPE == "hits":
        specs = build_hits_specs(NUM_VOLIDS)
    elif FILE_TYPE == "singles":
        specs = build_singles_specs(NUM_VOLIDS)
    else:
        specs = build_coinc_specs(NUM_VOLIDS)

    # Default to all fields if no USER_MASK provided
    mask = USER_MASK if USER_MASK is not None else [1] * len(specs)
    st, fields = build_struct_and_fields(specs, mask)

    # Parse and collect records
    records = list(parse_file(INFILE, st, fields))
    df = pd.DataFrame(records, columns=fields)

    # Write CSV
    os.makedirs(os.path.dirname(OUTCSV) or ".", exist_ok=True)
    df.to_csv(OUTCSV, index=False)
    print(f"Wrote {OUTCSV}: {len(df)} rows × {len(df.columns)} cols")