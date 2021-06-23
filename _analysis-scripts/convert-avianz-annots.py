## Convert AviaNZ JSON-format segment list
## into DCASE-style CSV timestamps.
## All subdirs in INDIR will be traversed
## and .data files summarized into OUTFILE.

## USAGE: python convert-avianz-annots.py INDIR OUTFILE

import csv
import os
import sys
sys.path.append('..')
import Segment

# INDIR = "dcase2018_baseline/task4/dataset/audio/eval/" or similar
# OUTFILE = any tmp csv file
INDIR = sys.argv[1]
OUTFILE = sys.argv[2]
print("Converting .data in dir", INDIR, "to file", OUTFILE)

# open the output csv file and print header
outfstream = open(OUTFILE, 'w', newline='')
csvf = csv.writer(outfstream, delimiter='\t')
csvf.writerow(["filename", "onset", "offset", "event_label"])

segments = Segment.SegmentList()
for root, dirs, files in os.walk(INDIR):
    for filename in files:
        if filename.endswith(".data"):
            print("found file", filename)

            # read JSON
            segments.parseJSON(os.path.join(root, filename))

            # for each segment in JSON,
            # create a CSV line
            for seg in segments:
                # Assuming here each segment has only 1 label!
                csvf.writerow([filename, seg[0], seg[1], seg[4][0]["species"]])

outfstream.close()
