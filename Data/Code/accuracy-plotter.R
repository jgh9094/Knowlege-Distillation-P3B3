# Can Plot the data for distillation runs
# Need to update folder and other parameters in ggplot function calls.

cat("\014")
setwd("C:/Users/josex/Desktop/Research/Projects/GEM-Summer-2021/Benchmarks/Pilot3/JGH/Data/Code/")
rm(list = ls())

library(ggplot2)
library(tidyverse)
library(cowplot)
library(RColorBrewer)
library(viridis)
library(Hmisc)
library(pastecs)
source("https://gist.githubusercontent.com/benmarwick/2a1bb0133ff568cbe28d/raw/fb53bd97121f7f9ce947837ef1a4c65a73bffb3f/geom_flat_violin.R")
cb_palette <- "Paired"

axis_title = 12
axis_text = 10
title_size = 14
legend_size = 12

p_theme <- theme(
  text = element_text(size = 10),
  axis.title.x = element_text(size = axis_title),
  axis.title.y = element_text(size = axis_title),
  axis.text = element_text(size = axis_text),
  axis.text.x = element_text(angle = 0, vjust = 0.5),
  legend.title = element_text(size = legend_size, hjust=0.5),
  legend.text = element_text(size = legend_size),
  legend.position = "right",
  plot.title = element_text( face = "bold", size = title_size, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  plot.margin = margin(10, 20, 10, 10))

# get the data
mods = read.csv('../276models.csv')
mods$model = '276m'
# mods$name = '276'
aggs = read.csv('../agg.csv')
# aggs$name = 'ensemble'
dist = read.csv('../01.csv')
# dist$name = 'distilled'
# dist = filter(dist, model == 't-1')
# stack the data
df <- rbind(mods,aggs,dist)


Beh_Mic = ggplot(df, aes(x = model, y = Beh_Mic, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Beh_Mic, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Behavior Micro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.975,.98))+
  p_theme
Beh_Mac = ggplot(df, aes(x = model, y = Beh_Mac, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Beh_Mac, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Behavior Macro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.84,.93))+
  p_theme

plot_grid(Beh_Mac, Beh_Mic,nrow = 2)
ggsave("01-behavior.pdf")



His_Mic = ggplot(df, aes(x = model, y = His_Mic, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = His_Mic, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Histology Micro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.77,.8))+
  p_theme
His_Mac = ggplot(df, aes(x = model, y = His_Mac, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = His_Mac, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Histology Macro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(0.3,.4))+
  p_theme

plot_grid(His_Mac, His_Mic,nrow = 2)
ggsave("01-histology.pdf")


Lat_Mic = ggplot(df, aes(x = model, y = Lat_Mic, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Lat_Mic, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Laterality Micro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.915,.925))+
  p_theme
Lat_Mac = ggplot(df, aes(x = model, y = Lat_Mac, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Lat_Mac, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Laterality Macro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(0.52,.57))+
  p_theme

plot_grid(Lat_Mac, Lat_Mic,nrow = 2)
ggsave("01-laterality.pdf")



Site_Mic = ggplot(df, aes(x = model, y = Site_Mic, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Site_Mic, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Site Micro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.925,.935))+
  p_theme
Site_Mac = ggplot(df, aes(x = model, y = Site_Mac, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Site_Mac, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Site Macro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(0.68,.72))+
  p_theme

plot_grid(Site_Mac, Site_Mic,nrow = 2)
ggsave("01-site.pdf")



Subs_Mic = ggplot(df, aes(x = model, y = Subs_Mic, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Subs_Mic, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Subsite Micro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(.68,.7))+
  p_theme
Subs_Mac = ggplot(df, aes(x = model, y = Subs_Mac, fill = model)) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .8) +
  geom_point(aes(y = Subs_Mac, color = model), position = position_jitter(width = .1), size = 1.5, alpha = 10.8) +
  geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.5) +
  xlab("Model Type") + ylab('Accuracy') + ggtitle("Subsite Macro Scores") +
  guides(fill = FALSE) +
  guides(color = FALSE) +
  # scale_color_brewer(palette = "Dark2") +
  # scale_fill_brewer(palette = "Dark2") +
  scale_y_continuous(limits=c(0.32,.37))+
  p_theme

plot_grid(Subs_Mac, Subs_Mic,nrow = 2)
ggsave("01-subsite.pdf")



















