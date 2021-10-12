## Analyse the results of LSK Zealandia survey:
## compare the raw and reviewed annotations obtained with various wind denoising methods
## and run SCR.
## (Table 1 in the paper)

options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(rjson)
library(rgdal)
library(ascr)

# set appropriately:
indir = "~/Documents/kiwis/wind/deposited/surveys/"

## SETTINGS AND HELPERS

# Reads all AviaNZ style annotations from directory dir,
# over recorders recs (string vector).
# Does some basic date conversions and returns a df.
readAnnots = function(dir, recs){
  annot = data.frame()
  for(rec in recs){
    gooddata = list.files(dir, pattern=paste0(rec, "_.*wav.data"), recursive=T)
    for(f in gooddata){
      a = fromJSON(file=paste(dir, f, sep="/"))
      if(length(a)>1){
        a = a[-1] # drop metadata
        a = data.frame(t(sapply(a, c))) # to dataframe
        tstamp = gsub("Z[A-Z]_(.*).wav.data", "\\1", f)
        a$time = parse_date_time(tstamp, "Ymd_HMS")
        a$rec = rec
        annot = rbind(annot, a)
      }
    }
    if(length(gooddata)==0) print(paste("Warning: no files found for recorder", rec))
  }
  if(nrow(annot)==0) print("Warning: no files read!")
  
  # actual start and end of call
  annot$start = annot$time + seconds(annot$X1)
  annot$end = annot$time + seconds(annot$X2)
  
  # JSON annotations are read in as list columns, so convert/drop those
  annot$species = unlist(lapply(annot$X5, function(x) x[[1]]$species))
  annot$calltype = unlist(lapply(annot$X5, function(x) x[[1]]$calltype))
  annot = annot[,6:11]
  
  annot$calllength = annot$end - annot$start
  
  return(annot)
}

# ggplot syntax for plotting raw & reviewed annotations.
# only prints one day, and expects a $start column
plotDayAnnots = function(df){
  if(min(day(df$start))==5 & max(day(df$start))==5){
    minbreaks = seq(ymd_hms("2018-11-05 22:00:00"), ymd_hms("2018-11-06 00:00:00"), by=60)
  } else if(min(day(df$start))==6 & max(day(df$start))==6) {
    minbreaks = seq(ymd_hms("2018-11-06 02:00:00"), ymd_hms("2018-11-06 04:00:00"), by=60)
  } else if(min(day(df$start))==10 & max(day(df$start))==10) {
    minbreaks = seq(ymd_hms("2018-11-10 22:00:00"), ymd_hms("2018-11-11 00:00:00"), by=60)
  } else {
    stop("Error: supplied multiple days")
  }
  
  p = ggplot(df) + geom_segment(aes(x=start, xend=end, y=rec, yend=rec, col=stage), size=5) +
    facet_wrap(~quarter, scales="free_x", nrow=4) +
    scale_color_manual(values=c("goldenrod1", "green4")) +
    scale_x_datetime(expand=c(0.01,0.01)) + 
    theme_minimal() + theme(panel.grid.major.y = element_blank(), strip.background = element_blank(),
                            strip.text = element_blank())
  print(p)
}

# Converts start-end type annotations into
# presence-absence over each second in the survey.
# Requires columns $daystart and $startadj, and $calllength.
convtoPresAbs = function(df){
  # create a vector of seconds along the entire survey
  df = mutate(df, relday=day(start)-min(day(start)))
  if(length(unique(df$relday))==1){
    # (subsequent lapply would be weird if using df here)
    timerange = list(seq(0, ceiling(difftime(max(df$endadj), min(df$startadj), units="secs"))))
  } else {
    timerange = group_by(df, relday) %>%
      summarize(numsecs = ceiling(difftime(max(endadj), min(startadj), units="secs"))) %>%
      mutate(numsecs = as.numeric(numsecs)) %>%
      apply(., 1, function(x) seq(0, x[2]))
  }
  
  # this is a list of dfs for each day:
  presence = lapply(seq_along(timerange), function(id) data.frame(secs = paste(id, timerange[[id]], sep="_"),
                                                                  ZA=0, ZC=0, ZE=0, ZG=0, ZH=0, ZI=0, ZJ=0, ZK=0))
  # rename the presence list to use actual days
  # (needed when there are gaps in reldays)
  names(presence) = sort(unique(df$relday))
  
  # convert start and end to relative times for that day
  df = group_by(df, relday) %>%
    mutate(start = as.numeric(startadj-min(startadj), units="secs")) %>%
    mutate(end = start + calllength)
  # Convert into 0/1 by second
  for(r in 1:nrow(df)){
    callstart = floor(df$start[r])
    callend = ceiling(df$end[r])
    day = as.character(df$relday[r])
    rec = df$rec[r]
    presence[[day]][callstart:callend, rec] = 1
  }
  # convert to a single df (flatten over days):
  presence = bind_rows(presence)
  return(presence)
}

# Converts to make.capt style capture history.
# From a presence-absence over each time and recorder,
# extracts a call detection matrix: callId x recorder,
# where callId is the start time of a contiguous presence region
identifyCalls = function(df){
  calls = data.frame()
  df = mutate(df, anycall = ZA | ZC | ZE | ZG | ZH | ZI | ZJ | ZK)
  i = 1
  while(i <= nrow(df)){
    ## find start of call
    if(df$anycall[i]){
      ## call detected on any of the recorders
      callTime = df$secs[i]
      j = i+1
      ## find when this call ends, assuming contiguous call across all recs
      while(df$anycall[j] & j<=nrow(df)){
        j = j+1
      }
      ## find recorders which had that call
      callRecs = names(which(sapply(df[i:j-1, 2:9], any)))
      for(rec in callRecs){
        calls = rbind(calls, data.frame(session=1, id=callTime, occ=1, rec=rec))
      }
      i = j
    }
    i = i+1
  }
  return(calls)
}


## READ IN ANNOTATIONS (raw and reviewed)
recs = c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK")
ans_rev_no = readAnnots(paste(indir, "reviewed", "CHP", "LSKnew", sep="/"), recs)
ans_rev_ols = readAnnots(paste(indir, "reviewed", "CHP-OLS", "LSKnew", sep="/"), recs)
ans_rev_qr = readAnnots(paste(indir, "reviewed", "CHP-QR", "LSK", sep="/"), recs)
ans_raw_no = readAnnots(paste(indir, "rawannots", "CHP", "LSKnew", sep="/"), recs)
ans_raw_ols = readAnnots(paste(indir, "rawannots", "CHP-OLS", "LSKnew", sep="/"), recs)
ans_raw_qr = readAnnots(paste(indir, "rawannots", "CHP-QR", "LSK", sep="/"), recs)

ans_rev_no$method = "none"
ans_raw_no$method = "none"
ans_rev_ols$method = "OLS"
ans_raw_ols$method = "OLS"
ans_rev_qr$method = "QR"
ans_raw_qr$method = "QR"
ans_rev_no$stage = "reviewed"
ans_raw_no$stage = "raw"
ans_rev_ols$stage = "reviewed"
ans_raw_ols$stage = "raw"
ans_rev_qr$stage = "reviewed"
ans_raw_qr$stage = "raw"

ans_all = bind_rows(ans_rev_no, ans_raw_no, ans_rev_ols, ans_raw_ols, ans_rev_qr, ans_raw_qr)
head(ans_all)
table(ans_all$method, ans_all$stage, day(ans_all$time))
table(ans_all$method, ans_all$stage)

# total num TP/FP seconds by method
group_by(ans_all, method, stage) %>%
  summarize(P=sum(calllength))

# attach this for plotting
ans_all = mutate(ans_all, quarter=pmin(3, ifelse(day(start)==5,
                                          difftime(start, ymd_hms("181105 22:00:00"), units="mins"),
                                          ifelse(day(start)==6,
                                          difftime(start, ymd_hms("181106 02:00:00"), units="mins"),
                                          difftime(start, ymd_hms("181110 22:00:00"), units="mins"))) %/% 30))

# clock adjustments, estimated using the CallCompare branch:
clockadjs = tibble(rec=c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK"), adjs=c(159,132,0, 82,102,31, 0,34))
ans_all = left_join(ans_all, clockadjs, by="rec") %>%
  mutate(startadj = start+adjs, endadj=end+adjs) %>%
  arrange(rec, stage)


## PLOT ALL CALLS
# times adjusted for clock drift
# NO wind adjustment:
ans_all %>%
  filter(method=="none", day(startadj)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="none", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="none", day(startadj)==10) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots

# wind-adjustment by OLS:
ans_all %>%
  filter(method=="OLS", day(startadj)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="OLS", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="OLS", day(startadj)==10) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots

# wind-adjustment by QR:
ans_all %>%
  filter(method=="QR", day(startadj)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="QR", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots


# equalize review effort (downsample methods with more reported detections)
dsfactor_no = nrow(ans_raw_ols)/nrow(ans_raw_no)
dsfactor_ols = nrow(ans_raw_ols)/nrow(ans_raw_ols)

# subset by time: from each 5 min window 04:00-04:05,04:05-04:10,..., take first X %
# where X is chosen to produce equal numbers of annotations for all methods.
# create a relative time counter from that day's start
ans_all = mutate(ans_all, secsfromstart=ifelse(day(start)==5,
                                        difftime(startadj, ymd_hms("181105 22:00:00"), units="secs"),
                                        ifelse(day(start)==6,
                                        difftime(startadj, ymd_hms("181106 02:00:00"), units="secs"),
                                        difftime(startadj, ymd_hms("181110 22:00:00"), units="secs"))))

ans_final_no = filter(ans_all, method=="none", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_no*300)
ans_final_ols = filter(ans_all, method=="OLS", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_ols*300)

# Same plots, but effort reduced to match best effort:
ans_all %>%
  filter(method=="none", day(startadj)==5) %>%
  filter(secsfromstart %% 300 < dsfactor_no*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="none", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_no*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots

ans_all %>%
  filter(method=="OLS", day(startadj)==5) %>%
  filter(secsfromstart %% 300 < dsfactor_ols*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots
ans_all %>%
  filter(method=="OLS", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_ols*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotDayAnnots


## CONVERT TO BINARY DETECTION MATRIX

# extract the right df with the adjusted times
presence_no = convtoPresAbs(ans_final_no)
presence_ols = convtoPresAbs(ans_final_ols)

# identify unique contiguous calls and their matches across recorders
calls_no = identifyCalls(presence_no)
calls_no

calls_ols = identifyCalls(presence_ols)
calls_ols

calls_qr = identifyCalls(presence_qr)
calls_qr

# number of calls x recorders
nrow(calls_no)  # 59
nrow(calls_ols) # 227

# we can see that the ratio enforced at annotation level matches ratio of call detections:
nrow(calls_no)/nrow(calls_ols); dsfactor_no
nrow(calls_ols)/nrow(calls_ols); dsfactor_ols


# ------- SCR -------

# GPS DATA:
# create recorder grid, using fixed distances for now
gpspos = read.table("./map_data/ZealandiaPoints.txt", sep=" ", nrows=10, h=F)
gpspos$V1 = paste0("Z", substr(gpspos$V1, 2, 2))
gpspos = filter(gpspos, V1 %in% recs)
ggplot(gpspos, aes(y=V2, x=V3)) + geom_point() + geom_text(aes(label=V1), nudge_y=0.0005) +
  theme(aspect.ratio = 1)

# Zealandia fence polygon, NZTM
fence = readOGR("./map_data/zealandia-polygon-merged.shp")

# project lat / long to easting / northing in meters, NZTM
coordinates(gpspos) = c("V3", "V2")
proj4string(gpspos) = CRS("+proj=longlat +datum=WGS84")
gpsposM = spTransform(gpspos,
                      CRS("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"))
gpsposM = data.frame(gpsposM)
colnames(gpsposM) = c("rec", "east", "north", "optional")

# center everything for simpler plotting later:
mapcentroidE = mean(gpsposM$east)
mapcentroidN = mean(gpsposM$north)
gpsposM = mutate(gpsposM, east=east-mapcentroidE, north=north-mapcentroidN, optional=NULL)
ggplot(gpsposM, aes(y=north, x=east)) +
  geom_point() + geom_text(aes(label=rec), nudge_y=50) +
  theme_minimal() + theme(aspect.ratio = 1)

# create trap coord matrix and the integration mask
scr_mask_bufsize = 1200
trapgrid = gpsposM %>% mutate(recid=1:n())
traps = as.matrix(trapgrid[,c("east", "north")])
mask = create.mask(traps, buffer=scr_mask_bufsize)

# crop mask to the inside of Zealandia.
# this requires converting the mask into SpatialPoints:
maskNonCentred = mask
maskNonCentred[,1] = maskNonCentred[,1] + mapcentroidE
maskNonCentred[,2] = maskNonCentred[,2] + mapcentroidN
maskDF = SpatialPoints(maskNonCentred, proj4string=CRS(proj4string(fence)))
maskOutside = is.na(over(maskDF, fence)$ID)

# plot before cropping:
plot(mask)
points(traps, col="red", pch=4)
points(mask[maskOutside,], col="seagreen")

# actually crop the mask:
mask = mask[!maskOutside,]


## RUNNING WITH UNADJUSTED CHP ANNOTS
# Create the call history
capt_no = inner_join(calls_no, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_no = create.capt(as.data.frame(capt_no), traps=traps)

# inspect the call history over recorders
capt_no$bincapt
nrow(capt_no$bincapt)
table(rowSums(capt_no$bincapt))

# Fit the model, accounting for the shorter duration
cr = fit.ascr(capt_no, traps, mask, survey.length = dsfactor_no)
summary(cr)

detfnparams_no = cr$coefficients
precision_no = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# Bootstrap to get good SE estimates
boot_no = boot.ascr(cr, N=500, n.cores=4)
summary(boot_no)
stdEr(boot_no, mce=T)  # SE(D) 0.55, SE(sigma) 21 w/ 1.5 nights
stdEr(boot_no)/coef(boot_no)*100 # CoV(D) 51.1 % w/ 1.5 nights
# Can use this to store boot'ed outputs:
# save(boot_no, file=paste0(indir, "LSK/", "boot_no.RData"))


## RUNNING WITH OLS-ADJUSTED ANNOTS
# Create the call history
capt_ols = inner_join(calls_ols, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_ols = create.capt(as.data.frame(capt_ols), traps=traps)

# inspect the call history over recorders
capt_ols$bincapt
nrow(capt_ols$bincapt)
table(rowSums(capt_ols$bincapt))

cr = fit.ascr(capt_ols, traps, mask, survey.length = dsfactor_ols)
summary(cr)

detfnparams_ols = cr$coefficients
precision_ols = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# Bootstrap to get good SE estimates
boot_ols = boot.ascr(cr, N=500, n.cores=4)
summary(boot_ols)
stdEr(boot_ols, mce=T) # SE(D) 0.125, SE(sigma) 11.3 
stdEr(boot_ols)/coef(boot_ols)*100 # CoV(D) 12.2 %
# Can use this to store boot'ed outputs:
# save(boot_ols, file=paste0(indir, "LSK/", "boot_ols.RData"))

