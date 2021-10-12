# Script for analysing and plotting the pilot data
# (Figures 3 and S1 in the paper, and AICcs)
library(seewave)
library(tuneR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)

# set these directories appropriately
OUTDIR = "~/Documents/gitrep/drafts/wind/"
INDIR = "~/Documents/kiwis/wind/deposited/pilotdata/"

wavfiles = list.files(INDIR, pattern="_2300.*wav$")

# Ideal passbands for the wavelet nodes
freqconvtable = data.frame(node=1:62,
                           fru = c(4000.0, 8000.0, 2000.0, 4000.0, 8000.0, 6000.0, 1000.0, 2000.0, 4000.0, 3000.0,
                                   8000.0, 7000.0, 5000.0, 6000.0, 500.0, 1000.0, 2000.0, 1500.0, 4000.0, 3500.0,
                                   2500.0, 3000.0, 8000.0, 7500.0, 6500.0, 7000.0, 4500.0, 5000.0, 6000.0, 5500.0,
                                   250.0, 500.0, 1000.0, 750.0, 2000.0, 1750.0, 1250.0, 1500.0, 4000.0, 3750.0,
                                   3250.0, 3500.0, 2250.0, 2500.0, 3000.0, 2750.0, 8000.0, 7750.0, 7250.0, 7500.0,
                                   6250.0, 6500.0, 7000.0, 6750.0, 4250.0, 4500.0, 5000.0, 4750.0, 6000.0, 5750.0, 5250.0, 5500.0),
                           frl = c(0.0, 4000.0, 0.0, 2000.0, 6000.0, 4000.0, 0.0, 1000.0, 3000.0, 2000.0, 7000.0,
                                   6000.0, 4000.0, 5000.0, 0.0, 500.0, 1500.0, 1000.0, 3500.0, 3000.0, 2000.0,
                                   2500.0, 7500.0, 7000.0, 6000.0, 6500.0, 4000.0, 4500.0, 5500.0, 5000.0,
                                   0.0, 250.0, 750.0, 500.0, 1750.0, 1500.0, 1000.0, 1250.0, 3750.0, 3500.0, 3000.0,
                                   3250.0, 2000.0, 2250.0, 2750.0, 2500.0, 7750.0, 7500.0, 7000.0, 7250.0, 6000.0,
                                   6250.0, 6750.0, 6500.0, 4000.0, 4250.0, 4750.0, 4500.0, 5750.0, 5500.0, 5000.0, 5250.0))
freqconvtable$frc = (freqconvtable$frl + freqconvtable$fru)/2


alles1 = alles2 = allpers = tibble() 
for(file in wavfiles){
  print(paste("working on file", file))
  # Energies are wavelet energies for each node,
  # except root, averaged over 0.1 s windows
  # read WPT energies made with dmey or sym8 wavelet
  efile1 = read.table(paste0(INDIR, file, "-dmey2.energies"), h=F)
  efile2 = read.table(paste0(INDIR, file, "-sym8.energies"), h=F)
  
  # read audio
  wav1 = readWave(paste0(INDIR, file), from=0, to=15, units="seconds")
  wav1 = ffilter(wav1, f=32000, to=8000)
  wav1 = resamp(wav1, f=32000, g=16000)
  
  # parse filename into components
  file = strsplit(file, "_")[[1]]
  rec = file[1]
  date = file[2]
  
  # select 3 windows from the first 15 s (verified animal free manually)
  # and 5th level nodes
  efile1 = data.frame(t(efile1[c(1,51,101), 31:62]), node=31:62, date=date, rec=rec)
  alles1 = bind_rows(alles1, efile1)
  
  # read WPT energies made with sym wavelet
  efile2 = data.frame(t(efile2[c(1,51,101), 31:62]), node=31:62, date=date, rec=rec)
  alles2 = bind_rows(alles2, efile2)
  
  # create periodograms
  ft1 = spec.pgram(wav1[(0*16000+1):(0.1*16000)], plot=F, spans=7)
  ft2 = spec.pgram(wav1[(5*16000+1):(5.1*16000)], plot=F, spans=7)
  ft3 = spec.pgram(wav1[(10*16000+1):(10.1*16000)], plot=F, spans=7)
  period1 = data.frame(X1=ft1$spec, X51=ft2$spec, X101=ft3$spec, frc=1600*ft1$freq, date=date, rec=rec)
  allpers = bind_rows(allpers, period1)
}

### -------------- Main plots ----------------

longdf = gather(alles1, key="time", value="wve", X1:X101) %>%
  left_join(freqconvtable[,c(1,4)], by="node") %>%
  mutate(rec=factor(rec, levels=c("ZI", "ZA", "KNR28"),
                    labels=c("windy", "sheltered", "AR4")))

pall = ggplot(longdf) +
  geom_line(aes(x=frc, y=log10(wve), group=paste(time, date, rec), col=date)) +
  scale_x_log10(limits=c(120, 7800)) +
  coord_cartesian(ylim=c(1.5,8.5)) + 
  facet_grid(.~rec) + 
  xlab("center freq., Hz") + ylab("log10(energy)") + 
  theme_grey() + theme(legend.position = "bottom", strip.text=element_text(size=12))

selecteddf = filter(alles1, date=="20210309" |
                      (rec=="ZA" & date=="20181115") |
                      (rec=="ZI" & date=="20190115")) %>%
  group_by(date) %>%
  left_join(freqconvtable[,c(1,4)], by="node")

p1 = ggplot(selecteddf, aes(x=frc, y=log10(X1), group=rec)) +
  geom_point(col="grey60", size=0.9) + geom_line(alpha=0.5, col="grey60") +
  scale_x_log10(limits=c(120, 7800)) +
  coord_cartesian(ylim=c(1.5,8.5)) + 
  geom_smooth(method="lm", col="purple", se=F, size=0.7) + 
  xlab(NULL) + ylab("log10(energy)") + 
  theme_bw() + theme(legend.position = "bottom")

p3 = ggplot(selecteddf, aes(x=frc, y=log10(X1), group=rec)) +
  geom_point(col="grey60", size=0.9) + geom_line(alpha=0.5, col="grey60") +
  scale_x_log10(limits=c(120, 7800)) +
  coord_cartesian(ylim=c(1.5,8.5)) + 
  geom_smooth(method="lm", formula=y~x+poly(x,3), col="purple", se=F, size=0.7) + 
  xlab("center freq., Hz") + ylab("log10(energy)") + 
  theme_bw() + theme(legend.position = "bottom")

pright = plot_grid(p1, p3, labels=c("B", "C"), nrow=2)
plot_grid(pall, pright, labels=c("A", ""), ncol=2, rel_widths=c(2,1))
ggsave(paste0(OUTDIR, "fig3-pilotdata.png"), width=8, height=4.5)

# Average slope over all
summary(lm(log(longdf$wve) ~ longdf$rec + longdf$rec*log(longdf$frc)))


### ----- alternative wavelets/periodogram --------

longdf2 = gather(alles2, key="time", value="wve", X1:X101) %>%
  left_join(freqconvtable[,c(1,4)], by="node") %>%
  mutate(rec=factor(rec, levels=c("ZI", "ZA", "KNR28"),
                    labels=c("windy", "sheltered", "AR4")))

pspecsym = ggplot(longdf2) +
  geom_line(aes(x=frc, y=log10(wve), group=paste(time, date, rec), col=date)) +
  scale_x_log10(limits=c(120, 7800)) +
  coord_cartesian(ylim=c(1.5,8.5)) + 
  facet_grid(.~rec) + 
  xlab(NULL) + ylab("log10(energy)") + 
  theme_grey() + theme(legend.position = "none", strip.text=element_text(size=12))

longdf3 = gather(allpers, key="time", value="wve", X1:X101) %>%
  mutate(frc=frc*10, wve=wve*6) %>%
  filter(frc%%6==0) %>% # downsample 6x roughly based on the Daniell kernel width
  mutate(rec=factor(rec, levels=c("ZI", "ZA", "KNR28"),
                    labels=c("windy", "sheltered", "AR4")))

pspecper = ggplot(longdf3) +
  geom_line(aes(x=frc, y=log10(wve), group=paste(time, date, rec), col=date)) +
  scale_x_log10(limits=c(120, 7800)) +
  coord_cartesian(ylim=c(-3.5,2)) + 
  facet_grid(.~rec) + 
  xlab("center freq., Hz") + ylab("log10(energy)") + 
  theme_grey() + theme(legend.position = "bottom", strip.text=element_text(size=12))

plot_grid(pspecsym, pspecper, nrow=2, rel_heights = c(7,10))
ggsave(paste0(OUTDIR, "figs1-pilotdata-supp.png"), width=8, height=6.5)


### ------------------ AIC  -----------------------
# AIC corrected for small sample size, credit to gamlr package
AICc = function(object, k = 2){
  ll <- logLik(object)
  d <- attributes(ll)$df
  n <- attributes(ll)$nobs
  ic <- -2 * ll + k * d * n/(n - d - 1)
  ic[d + 1 > n] <- Inf
  attributes(ic)[c("df", "nobs", "class")] <- NULL
  ic
}

# fit linear model to each, average AICc
get_aic_rsq = function(y, x){
  mod = lm(y ~ x)
  aic_lin = AICc(mod)
  rsq_lin = summary(mod)$r.squared
  mod = lm(y ~ poly(x, 2))
  aic_2 = AICc(mod)
  mod = lm(y ~ poly(x, 3))
  aic_3 = AICc(mod)
  mod = lm(y ~ poly(x, 4))
  aic_4 = AICc(mod)
  mod = lm(y ~ poly(x, 5))
  aic_5 = AICc(mod)
  mod = lm(y ~ poly(x, 6))
  aic_6 = AICc(mod)
  return(data.frame(rsq_lin, aic_lin, aic_2, aic_3, aic_4, aic_5, aic_6))
}
aics_perclip = group_by(longdf, rec, date, time) %>%
  filter(frc<=6000, frc>=150) %>%  # remove non-wind freqs
  mutate(logE = log(wve), logF = log(frc)) %>%
  summarize(get_aic_rsq(logE, logF))

aics_perclip %>%
  summarize(a1 = mean(aic_lin), a2=mean(aic_2), a3=mean(aic_3), a4=mean(aic_4), a5=mean(aic_5), a6=mean(aic_6))

aics_perclip %>%
  ungroup() %>%
  summarize(a1 = mean(aic_lin), a2=mean(aic_2), a3=mean(aic_3), a4=mean(aic_4), a5=mean(aic_5), a6=mean(aic_6))
