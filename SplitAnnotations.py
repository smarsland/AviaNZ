# Companion script to SplitWav audio splitter.
# Splits AviaNZ-format annotation files.

import json, copy
import datetime as dt

# INPUT ARGS:
t = 60 # split chunk size, seconds
infile = '/media/julius/unishare/Zealandia_annotations_RH/ZJ_20180912_180522.wav.data'
outprefix = '/tmp/python/ZJ_' # directory + filestem. Will be suffixed with "YMD_HMS.wav.data"


infilestem = infile.split(".")[-2] # drop extension
datestamp = infilestem.split("_")[-2:] # get [date, time]
outtime = '_'.join(datestamp) # make "date_time"


file = open(infile, 'r')
segs = json.load(file)
header = s[0]
body = s[1:]

d = dt.datetime.strptime(outtime, "%Y%m%d_%H%M%S")

maxtime = max([seg[1] for seg in body])

# repeat initial meta-segment for each output file
all = [[header] for i in range(maxtime // t + 1)]

# separate segments into output files and adjust segment timestamps
for b in body:
    filenum, adjst = divmod(b[0], t)
    adjend = b[1] % t 
    if b[1] > (filenum+1)*t:
        adjend = adjend + t 
    # print(filenum)
    # print(all[int(filenum)])
    all[int(filenum)].append([adjst, adjend, b[2], b[3], b[4]])

# save files, while increasing the filename datestamps
for a in all:
    f2 = open(str(outprefix) + dt.datetime.strftime(d, "%Y%m%d_%H%M%S") + '.wav.data', 'w')
    json.dump(a, f2) 
    f2.close()
    d = d + dt.timedelta(seconds=t)

