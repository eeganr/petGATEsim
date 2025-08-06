import struct
import pandas as pd
import os
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="aug1flange/output", help="path prefix for input files")
parser.add_argument("-o", "--output", type=str, default="aug1dfs/output", help="path prefix for output files")
parser.add_argument("-s", "--start", type=int, default=1, help="start file num")
parser.add_argument("-e", "--end", type=int, default=100, help="end file num")


args = parser.parse_args()

NUM_VOLIDS = 6
FILE_RANGE = range(args.start, args.end + 1)
PATH_PRE_PREFIX = '/scratch/users/eeganr/'
PATH_PREFIX = PATH_PRE_PREFIX + args.path
OUT_PREFIX = PATH_PRE_PREFIX + args.output

def read_bin_file(infile, outfile):
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
            ("phantomCom","s", 8),
            ("phantomRay","s", 8),
        ]


    # Convert specs + mask into struct and field names
    def build_struct_and_fields(specs, mask):
        if len(mask) != len(specs):
            raise ValueError(f"Mask length {len(mask)} != number of specs {len(specs)}")
        fmt = "<"
        fields = []
        for keep, (name, fchar, cnt) in zip(mask, specs):
            if keep:
                fmt += f"{cnt}{fchar}"
                if cnt == 1 or fchar=='s':
                    fields.append(name)
                else:
                    for idx in range(cnt):
                        fields.append(f"{name}_{idx}")
        return struct.Struct(fmt), fields


    def parse_file(path, st, fields):
        with open(path, "rb") as f:
            count = 0
            filesize = os.path.getsize(path)

            assert filesize % st.size == 0

            for i in range(filesize // st.size):
                data = f.read(st.size)
                record = st.unpack(data)
                if (count % 1e6 == 0):
                    print(f'Read {int(count // 1e6)} million records.')
                if (i == 15e6):
                    break
                count += 1
                yield record

            

    specs = build_singles_specs(NUM_VOLIDS)
    mask = [1] * len(specs)
    st, fields = build_struct_and_fields(specs, mask)
    print(f"Reading file {infile}...")
    records = list(parse_file(infile, st, fields))
    print(f"Parsed {len(records)} records!")


    singles = pd.DataFrame({
        "time": [record[12] for record in records],
        "source": [record[3:6] for record in records],
        "energy": [record[13] for record in records],
        "detpos": [record[14:17] for record in records],
    })

    print('Made DF, dumping!')

    with open(outfile, 'wb') as f:
        pickle.dump(singles, f)

    print('Dumped!')


if __name__ == "__main__":
    for i in FILE_RANGE:
        # Step 1: Read file
        infile = PATH_PREFIX + f'{i}Singles.dat'
        outfile = OUT_PREFIX + f'{i}.pkl'
        singles = read_bin_file(infile, outfile)