# Script for plotting the chi squared distributions
# after different spectral subtractions
# (Figure 2 in the paper)
library(tidyr)
library(dplyr)
library(ggplot2)

# set appropriately
OUTDIR = "~/Documents/kiwis/wind/"

n = 500
varN1 = 3
varN2 = 5  # ratio of noisy2/noisy1
xnoisy1 = rchisq(n, 1) * varN1
xnoisy2 = rchisq(n, 1) * varN2
linsub1 = pmax(0, xnoisy1 - varN1)
linsub2 = pmax(0, xnoisy2 - varN2)
logsub1 = pmax(0, xnoisy1 / varN1)
logsub2 = pmax(0, xnoisy2 / varN2)

df = data.frame(x=c(xnoisy1, xnoisy2, linsub1, linsub2, logsub1, logsub2),
                distr=rep(c("A: noisy", "B: lin. sub.", "C: log. sub."), each=2*n),
                varN=rep(rep(c(varN1, varN2), each=n), 3))

# percents marked at lambda = 0.05 quantile
lambda = qchisq(0.95, 1)
markedA1 = 1-pchisq(lambda/varN1, 1)
markedA2 = 1-pchisq(lambda/varN2, 1)
markedB1 = 1-pchisq((lambda+varN1)/varN1, 1)
markedB2 = 1-pchisq((lambda+varN2)/varN2, 1)
markedC = 1-pchisq(lambda, 1)

fracdf = expand.grid(distr=c("A: noisy", "B: lin. sub.", "C: log. sub."),
                     varN=c(varN1, varN2))
fracdf$marked = c(markedA1, markedB1, markedC, markedA2, markedB2, markedC)
fracdf$marktext = paste(format(fracdf$marked*100, digits=3), "%")

df %>%
    ggplot(aes(x=as.character(varN), y=x)) +
    geom_violin(col="purple") +
    geom_jitter(size=0.2, width=0.01) +
    geom_hline(yintercept=lambda, col="darkorange") +
    geom_text(data=fracdf, angle=35, nudge_x=0.35,
            aes(x=as.character(varN), y=10, label=marktext), col="darkorange3") +
    facet_grid(.~distr) +
    theme_bw() + xlab("noise variance") + ylab(NULL) +
    theme(panel.grid.major.x=element_blank(),
          panel.grid.minor.y=element_blank(), strip.text=element_text(size=12))

ggsave(paste0(OUTDIR, "fig2-chisq.eps", width=6, height=2.7)

