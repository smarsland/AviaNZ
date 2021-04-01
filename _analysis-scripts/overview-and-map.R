## Makes initial maps and overviews of available WAV files for each of the two surveys
## Internal helper script - parses LINZ data (crops maps and merges shapefiles)
## which is not distributed in this repository.

options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rgdal)
library(lubridate)
library(raster)
library(rasterVis)
library(rgeos)
library(maptools)

# Lake Ellesmere bittern data (original, full) and maps from LINZ
indir = "~/Documents/audiodata/Bittern/LakeEllesmere/"
mapdir = "~/Documents/kiwis/soundchppaper/p3surveys/"

alltimes=tibble()
for(rec in paste0("BIT", 1:9)){
  print(rec)
  fs = list.files(paste0(indir, rec, "/Low/"), recursive=T, pattern=".*wav")
  fs2 = list.files(paste0(indir, rec, "/survey/"), recursive=T, pattern=".*wav")
  # strip dirnames and formats
  fs = gsub(".*/(.*).wav", "\\1", fs)
  fs2 = gsub(".*/(.*).wav", "\\1", fs2)
  newtimes=tibble()
  if(length(fs)){
    tryCatch({newtimes = tibble(times=parse_date_time(fs, "Ymd_HMS", tz="NZ"),
                                fname = "Low", rec=rec)}, 
             warning=function(w) print(list(w, fs)))
    alltimes = bind_rows(alltimes, newtimes)
  }
  newtimes=tibble()
  if(length(fs2)){
    tryCatch({newtimes = tibble(times=parse_date_time(fs2, "dmy_HMS", tz="NZ"),
                                 fname = "survey", rec=rec)}, 
            warning=function(w) print(list(w, fs)))
    alltimes = bind_rows(alltimes, newtimes)
  }
}
ggplot(alltimes, aes(x=times, y=rec, col=fname)) + geom_point(pch="|", size=4) +
  scale_x_datetime() + theme_bw()



### MAPPING
## Data from LINZ in NZTM2000
MAPLOCATION = paste0(mapdir, "map_photo/canterbury-03m-rural-aerial-photos-2015-2016.tif")

# read in map
gpsmap = readGDAL(MAPLOCATION)
mapprojection =  proj4string(gpsmap)
bbox(gpsmap)

# lake shapefile from LINZ, NZTM2000
lake = readOGR(paste0(mapdir, "map_photo/lake-polygon-hydro-190k-1350k.shp"))
lake = fortify(lake, region="fidn")
ggplot(lake, aes(x=long, y=lat)) +
  geom_polygon()

## 2. GPS DATA
# create recorder grid, using fixed distances for now
gpspos = read.table(paste0(mapdir, "recordergps.txt"), sep="\t", h=T)
ggplot(gpspos, aes(y=latitude, x=longitude)) + geom_point() + geom_text(aes(label=desc), nudge_y=0.0001) +
  theme(aspect.ratio = 1)

# Project rec positions to easting/northing
coordinates(gpspos) = c("longitude", "latitude")
proj4string(gpspos) = CRS("+proj=longlat +datum=WGS84")
gpspos = spTransform(gpspos, mapprojection)

trapbounds = bbox(gpspos)
trapbounds[,2] - trapbounds[,1] # 120 x 365 m section
# so add ~100 m buffer:
trapbounds = trapbounds + matrix(c(-100, 150,  -100, 100), nrow=2, byrow=T)   # W, E,  S, N
trapbounds

# get offset in m from start
mapgridparams = gridparameters(gpsmap)
trapbounds = trapbounds - mapgridparams$cellcentre.offset
# get offset in pixels from start
trapbounds = round(trapbounds/mapgridparams$cellsize)

# now, trapbounds y is pixels from min lattitude (bottom), but spatial df y-dim has top at 0.
# So, invert trapbounds y:
trapbounds[2,] = mapgridparams$cells.dim[2] - trapbounds[2,]

# extract subset
gpsmap = gpsmap[trapbounds[2,1]:trapbounds[2,2], trapbounds[1,1]:trapbounds[1,2]]
writeGDAL(gpsmap, paste0(mapdir, "map_photo/subset.tif"), drivername="GTiff", type="Byte")

bbox(gpsmap)
bbox(gpspos)

# speaker positions (in NZ TM already)
speakerpos = read.table(paste0(mapdir, "speakergps.txt"), sep="\t", h=T)

# final conversions for 12-07 data and plotting
gpsposDF = filter(data.frame(gpspos), !desc %in% c("BIT7", "BIT9"))
colnames(gpsposDF) = c("rec", "east", "north", "optional")
write.table(gpsposDF, paste0(mapdir, "recordergps_NZTM.txt"))
speakerposDF = data.frame(speakerpos)
colnames(speakerposDF) = c("east", "north", "rec")

# convert RGB to a colortable and assign it
gpsmapcols = SGDF2PCT(gpsmap[,,1:3])
gpsmap = raster(gpsmap)
gpsmap = setValues(gpsmap, gpsmapcols$idx)
colortable(gpsmap) = gpsmapcols$ct

# save it - note the maxpixels arg!!
# gpsmap = crop(gpsmap, extent(1745150, 1746000, 5425150, 5426200))   # Can use this to crop manually
p = gplot(gpsmap, maxpixels=8e6) +
  geom_raster(aes(fill=value)) +
  geom_point(aes(x=east, y=north), color="white", pch=4, data=gpsposDF) +
  geom_point(aes(x=east, y=north), color="orange", pch=16, data=speakerposDF) +
  geom_text(aes(x=east, y=north, label=rec), color="white", data=gpsposDF, vjust=0, nudge_y=10) +
  geom_polygon(aes(x=long, y=lat), fill="seagreen", data=lake) +
  xlab("Easting, m") + ylab("Northing, m") + 
  scale_fill_gradientn(colours=colortable(gpsmap), guide="none") +
  coord_fixed(expand=F, xlim=bbox(gpsmap)[1,], ylim=bbox(gpsmap)[2,]) + theme_bw()
p
ggsave(plot=p, filename=paste0(mapdir, "BitternLE/recorder_map.tiff"), device = "tiff", dpi=300)
ggsave(plot=p, filename=paste0(outdir, "recorder_map_Bittern.png"), dpi=300)



# Zealandia LSK maps from LINZ
mapdir = "~/Documents/kiwis/soundchppaper/p3surveys/"
outdir = "~/Documents/kiwis/soundchppaper/p3surveys/LSK/"

## PLOT THE AERIAL + RECORDER MAP
# GPS DATA:
# create recorder grid, using fixed distances for now
gpspos = read.table(paste0(mapdir, "ZealandiaPoints.txt"), sep=" ", nrows=10, h=F)
recs = c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK")
gpspos$V1 = paste0("Z", substr(gpspos$V1, 2, 2))
gpspos = filter(gpspos, V1 %in% recs)
ggplot(gpspos, aes(y=V2, x=V3)) + geom_point() + geom_text(aes(label=V1), nudge_y=0.0005) +
  theme(aspect.ratio = 1)

# project lat / long to easting / northing in meters, NZTM
coordinates(gpspos) = c("V3", "V2")
proj4string(gpspos) = CRS("+proj=longlat +datum=WGS84")
gpsposM = spTransform(gpspos,
                      CRS("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"))
gpsposM = data.frame(gpsposM)
colnames(gpsposM) = c("rec", "east", "north", "optional")

# MAP AERIAL:
gpsmap = readGDAL(paste0(mapdir, "map_photo/BQ31_subset.tif"))
bbox(gpsmap)
bbox(spTransform(gpspos, CRS(proj4string(gpsmap))))

# make sure projections match
proj4string(gpsmap)=="+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

# convert RGB to a colortable and assign it
gpsmapcols = SGDF2PCT(gpsmap[,,1:3])
gpsmap = raster(gpsmap)
gpsmap = setValues(gpsmap, gpsmapcols$idx)
colortable(gpsmap) = gpsmapcols$ct

# save it - note the maxpixels arg!!
gpsmap = crop(gpsmap, extent(1745150, 1746200, 5425250, 5426250))
p = gplot(gpsmap, maxpixels=8e6) +
  geom_raster(aes(fill=value)) +
  geom_point(aes(x=east, y=north), color="white", pch=4, data=gpsposM) +
  geom_text(aes(x=east, y=north, label=rec), color="white", data=gpsposM, vjust=0, nudge_y=20) +
  xlab("Easting, m") + ylab("Northing, m") + 
  scale_fill_gradientn(colours=colortable(gpsmap), guide="none") +
  coord_fixed(xlim=c(1745150, 1746200), ylim=c(5425250, 5426250), expand=F) + theme_bw()
p
ggsave(plot=p, filename=paste0(outdir, "recorder_map.tiff"), device = "tiff", dpi=300)
ggsave(plot=p, filename=paste0(outdir, "recorder_map_LSK.png"), dpi=300)


## MERGE TWO ZEALANDIA POLYGONS INTO ONE, NZTM
fence = readOGR(paste0(mapdir, "map_photo/zealandia-polygon-manual.shp"))
proj4string(fence)
fence = unionSpatialPolygons(fence, 1)
# project to NZTM
fence = spTransform(fence,
                    CRS("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"))
shapefile(fence, file=paste0(mapdir, "map_photo/zealandia-polygon-merged.shp"))

# sanity check
fence = fortify(fence, region="ID")
ggplot(gpsposM, aes(y=north, x=east)) +
  geom_point() + geom_text(aes(label=rec), nudge_y=50) +
  geom_polygon(data=fence, aes(x=long-mapcentroidE, y=lat-mapcentroidN), alpha=0.1, fill="green") +
  theme_minimal() + theme(aspect.ratio = 1) +
  coord_cartesian(xlim=c(-1000, 1000), ylim=c(-1000,1000))
