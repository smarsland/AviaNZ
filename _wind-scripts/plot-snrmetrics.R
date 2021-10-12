# Plotting the SNR metrics (barplots) for the denoising experiment
# from the summary csvs produced by measure-snr.py.
# (Figure 4 and Figure S2)

library(ggplot2)
library(dplyr)
library(tidyr)

# Set appropriately:
INDIR = "~/Documents/kiwis/wind/deposited/denoising/"
OUTDIR = "~/Documents/gitrep/drafts/wind/"
setwd(INDIR)

# ------------

metr = read.table("out_stats_mixed.csv", h=T,sep=",")
metr$mixingsnr = as.character(metr$mixingsnr)
metr$noisetype = as.character(metr$noisetype)

# thr const 3.0
metr2 = read.table("out_stats_dnconst3.csv", h=T,sep=",")
metr2$mixingsnr = as.character(metr2$mixingsnr)
metr2$noisetype = as.character(metr2$noisetype)

# thr ols 1.0
metr3 = read.table("out_stats_dnols1.csv", h=T,sep=",")
metr3$mixingsnr = as.character(metr3$mixingsnr)
metr3$noisetype = as.character(metr3$noisetype)

# thr qr 1.0
metr4 = read.table("out_stats_dnqr1.csv", h=T,sep=",")
metr4$mixingsnr = as.character(metr4$mixingsnr)
metr4$noisetype = as.character(metr4$noisetype)


head(metr)
# snr
ggplot(metr, aes(x=mixingsnr, y=log(snr))) +
  geom_point(aes(col=cleanf)) +
  facet_wrap(~noisetype) +
  theme_bw()

# sisdr
ggplot(metr, aes(x=mixingsnr, y=sisdr)) +
  geom_point(aes(col=cleanf)) +
  facet_wrap(~noisetype) +
  theme_bw()

metr_changes = bind_rows("dnconst"=metr2, "dnols"=metr3, "dnqr"=metr4, .id="trt") %>%
  inner_join(metr, by=c("mixingsnr", "noisetype", "noisyf", "cleanf", "sigtype"), suffix=c(".dn", ".n")) %>%
  mutate(impr.snr=10*log10(snr.dn)-10*log10(snr.n), impr.sisdr=sisdr.dn-sisdr.n) %>%
  mutate(mixingsnr = factor(mixingsnr, labels=c("+12 dB", "0 dB", "-12 dB")))
# SNRs are +12/0/-12 for handheld, +12/0 for soundsc.

# (for diagnostics)
ggplot(metr_changes, aes(x=mixingsnr, y=impr.snr)) +
  geom_point(aes(col=sigtype)) +
  facet_grid(noisetype~trt) +
  theme_bw()
ggplot(metr_changes, aes(x=mixingsnr, y=impr.sisdr)) +
  geom_point(aes(col=cleanf)) +
  facet_grid(noisetype~trt) +
  theme_bw()

## --------- outputs
# Main
metr_changes %>%
  mutate(sigtype=factor(sigtype, labels=c("Xeno-canto", "Soundscapes"))) %>%
  group_by(mixingsnr, sigtype, trt) %>%
  summarize(mean.impr=mean(impr.snr), se=sd(impr.snr)/sqrt(n()), n=n()) %>%
  ggplot(aes(x=mixingsnr, y=mean.impr)) +
  geom_col(aes(fill=trt, group=trt), col="grey10", position=position_dodge(), width=0.7) +
  geom_linerange(aes(group=trt, ymin=mean.impr-se, ymax=mean.impr+se),
                position=position_dodge(width=0.7)) + 
  scale_fill_manual(values=c("khaki", "#74C476", "#238B45"), name="Denoising thr.",
                    labels=c("constant", "OLS est.", "QR est.")) + 
  geom_hline(yintercept=0, col="red") +
  xlab("Mixing SNR") + ylab("Mean SNR improvement") +
  facet_wrap(~sigtype, scales="free_x") +
  theme_bw()
ggsave(paste0(OUTDIR,"snr-barplot-main.png"), width=5.5, height=3)

# Supplementary:
metr_changes %>%
  mutate(sigtype=factor(sigtype, labels=c("Xeno-canto", "Soundscapes"))) %>%
  group_by(mixingsnr, sigtype, trt) %>%
  summarize(mean.impr=mean(impr.sisdr), se=sd(impr.sisdr)/sqrt(n()), n=n()) %>%
  ggplot(aes(x=mixingsnr, y=mean.impr)) +
  geom_col(aes(fill=trt, group=trt), col="grey10", position=position_dodge(), width=0.7) +
  geom_linerange(aes(group=trt, ymin=mean.impr-se, ymax=mean.impr+se),
                 position=position_dodge(width=0.7)) + 
  scale_fill_manual(values=c("khaki", "#74C476", "#238B45"), name="Denoising thr.",
                    labels=c("constant", "OLS est.", "QR est.")) + 
  geom_hline(yintercept=0, col="red") +
  xlab("Mixing SNR") + ylab("Mean SI-SDR improvement") +
  facet_wrap(~sigtype, scales="free_x") +
  theme_bw()
ggsave(paste0(OUTDIR,"snr-barplot-supp.png"), width=5.5, height=3)

