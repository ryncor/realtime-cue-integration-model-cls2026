#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))

ROOT <- normalizePath('.', mustWork = TRUE)
PY <- file.path(ROOT, 'scripts', 'extract_control_revision_trace.py')
MODEL <- file.path(ROOT, 'demos', 'output', 'learning', 'phonetic_full_inventory_model.npz')
FIG <- file.path(ROOT, 'figures')
dir.create(FIG, showWarnings = FALSE)

if (!file.exists(PY)) stop('Missing ', PY)
if (!file.exists(MODEL)) stop('Missing ', MODEL)

FONT <- 'Noto Sans'

# Figure 1: single-tracker control automaton
node_df <- data.frame(
  name = c('IDLE', 'EVAL', 'COMM', 'REPROC'),
  x = c(0.85, 2.85, 4.85, 3.85),
  y = c(2.10, 2.10, 2.10, 0.64),
  stringsAsFactors = FALSE
)

p_auto <- ggplot() +
  geom_segment(
    aes(x = 1.28, xend = 2.42, y = 2.10, yend = 2.10),
    linewidth = 0.72,
    arrow = arrow(length = unit(0.17, 'cm'), type = 'closed')
  ) +
  geom_segment(
    aes(x = 3.28, xend = 4.42, y = 2.10, yend = 2.10),
    linewidth = 0.72,
    arrow = arrow(length = unit(0.17, 'cm'), type = 'closed')
  ) +
  geom_curve(
    aes(x = 2.64, y = 2.49, xend = 3.06, yend = 2.49),
    curvature = -1.00, linewidth = 0.72,
    arrow = arrow(length = unit(0.15, 'cm'), type = 'closed')
  ) +
  geom_curve(
    aes(x = 4.64, y = 2.49, xend = 5.06, yend = 2.49),
    curvature = -1.00, linewidth = 0.72,
    arrow = arrow(length = unit(0.15, 'cm'), type = 'closed')
  ) +
  # CONF -> REPROC
  geom_curve(
    aes(x = 4.54, y = 1.78, xend = 4.08, yend = 0.97),
    curvature = 0.24, linewidth = 0.72,
    arrow = arrow(length = unit(0.16, 'cm'), type = 'closed')
  ) +
  # REPROC -> CONF
  geom_curve(
    aes(x = 4.26, y = 0.77, xend = 4.97, yend = 1.67),
    curvature = 0.24, linewidth = 0.72,
    arrow = arrow(length = unit(0.16, 'cm'), type = 'closed')
  ) +
  geom_point(
    data = node_df,
    aes(x = x, y = y),
    inherit.aes = FALSE,
    shape = 21, size = 29, stroke = 0.78,
    fill = 'white', color = 'grey15'
  ) +
  geom_text(
    data = node_df,
    aes(x = x, y = y, label = name),
    inherit.aes = FALSE,
    family = FONT, fontface = 'bold', size = 4.2,
    color = 'grey10'
  ) +
  annotate('text', x = 1.85, y = 2.43, label = 'evidence arrives', family = FONT, size = 3.65) +
  annotate('text', x = 3.85, y = 2.43, label = 'commit', family = FONT, size = 3.65) +
  annotate('text', x = 2.85, y = 3.08, label = 'update', family = FONT, size = 3.65) +
  annotate('text', x = 4.85, y = 3.08, label = 'maintain', family = FONT, size = 3.65) +
  annotate('text', x = 4.02, y = 1.53, label = 'conflict', family = FONT, size = 3.55, hjust = 1) +
  annotate('text', x = 5.08, y = 1.18, label = 'revise', family = FONT, size = 3.55, hjust = 0) +
  coord_cartesian(xlim = c(0.12, 5.95), ylim = c(0.06, 3.30), clip = 'off') +
  theme_void(base_family = FONT) +
  theme(plot.margin = margin(6, 8, 6, 8))

ggsave(
  file.path(FIG, 'single_tracker_control_automaton.pdf'),
  p_auto,
  width = 9.4, height = 4.05,
  device = cairo_pdf, family = FONT
)

# Figure 2: uncluttered grayscale revision trace (three vertically aligned tiers)
trace_file <- tempfile(fileext = '.csv')
status <- system2('python3', c(PY, '--model', MODEL, '--out', trace_file),
                  stdout = '', stderr = '')
if (status != 0 || !file.exists(trace_file))
  stop('Failed to extract single-trial revision trace')
D <- read.csv(trace_file, stringsAsFactors = FALSE, check.names = FALSE)
unlink(trace_file)
needed <- c('t_ms', 'p_s', 'p_sh', 'state', 'boundary_ms',
             'speaker_available_ms', 'cue_cog', 'cue_spread', 'cue_skewness', 'cue_duration', 'cue_voicing', 'cue_f2_onset',
             'duration_support', 'vowel_support', 'release_support')
if (!all(needed %in% names(D)))
  stop('Trace extractor missing columns: ', paste(setdiff(needed, names(D)), collapse=', '))

XMIN <- min(D$t_ms)
XMAX <- max(D$t_ms) + 5
XBREAKS <- seq(0, floor(XMAX / 50) * 50, by = 50)
ONSET <- D$boundary_ms[1]
TALKER <- D$speaker_available_ms[1]

# All six measurements from the online trace, together on ONE common-axis plot.
cue_order <- c('Spectral center', 'Spectral spread', 'Spectral skewness',
               'Elapsed frication', 'Voicing proportion', 'Vowel F2 onset')
cues <- rbind(
  data.frame(t_ms=D$t_ms, cue='Spectral center', value=D$cue_cog),
  data.frame(t_ms=D$t_ms, cue='Spectral spread', value=D$cue_spread),
  data.frame(t_ms=D$t_ms, cue='Spectral skewness', value=D$cue_skewness),
  data.frame(t_ms=D$t_ms, cue='Elapsed frication', value=D$cue_duration),
  data.frame(t_ms=D$t_ms, cue='Voicing proportion', value=D$cue_voicing),
  data.frame(t_ms=D$t_ms, cue='Vowel F2 onset', value=D$cue_f2_onset)
)
cues$cue <- factor(cues$cue, levels=cue_order)
cue_greys <- c('Spectral center'='#121212', 'Spectral spread'='#454545',
               'Spectral skewness'='#1e1e1e', 'Elapsed frication'='#444444',
               'Voicing proportion'='#666666', 'Vowel F2 onset'='#555555')
cue_styles <- c('Spectral center'='solid', 'Spectral spread'='longdash',
                'Spectral skewness'='dotted', 'Elapsed frication'='dashed',
                'Voicing proportion'='dotdash', 'Vowel F2 onset'='twodash')
cue_shapes <- c('Spectral center'=16, 'Spectral spread'=17,
                'Spectral skewness'=15, 'Elapsed frication'=18,
                'Voicing proportion'=0, 'Vowel F2 onset'=1)

marker_offset <- match(cues$cue, cue_order) * 5
cue_points <- cues[!is.na(cues$value) &
                   ((cues$t_ms + marker_offset) %% 40 == 0), , drop=FALSE]

base_style <- theme_classic(base_size=11, base_family=FONT) +
  theme(plot.margin=margin(4, 16, 4, 25),
        legend.title=element_blank(),
        legend.text=element_text(size=9.2),
        axis.title.x=element_blank(),
        panel.grid.major.y=element_line(color='grey92', linewidth=.25),
        axis.title.y=element_text(size=10.2, margin=margin(r=7)))
markers <- function() list(
  geom_vline(xintercept=ONSET, color='grey60', linewidth=.45, linetype='dashed'),
  geom_vline(xintercept=TALKER, color='grey45', linewidth=.45, linetype='dotted')
)
x_axis <- scale_x_continuous(limits=c(XMIN,XMAX), breaks=XBREAKS,
                             expand=expansion(mult=c(0,0)))

p_cues <- ggplot(cues, aes(t_ms, value, color=cue, linetype=cue, group=cue)) +
  geom_hline(yintercept=0, color='grey82', linewidth=.32) +
  geom_step(linewidth=1.05, direction='hv', na.rm=TRUE) +
  geom_point(data=cue_points, aes(shape=cue), size=1.9, stroke=.75,
             show.legend=TRUE) +
  markers() +
  scale_color_manual(values=cue_greys, breaks=cue_order) +
  scale_linetype_manual(values=cue_styles, breaks=cue_order) +
  scale_shape_manual(values=cue_shapes, breaks=cue_order) +
  guides(color=guide_legend(nrow=2, byrow=TRUE,
                            override.aes=list(linewidth=1.35, size=2)),
         linetype=guide_legend(nrow=2, byrow=TRUE),
         shape=guide_legend(nrow=2, byrow=TRUE)) +
  labs(y='Cue value') + x_axis + base_style +
  theme(legend.position='bottom',
        legend.justification='center',
        legend.key.width=unit(1.25,'cm'),
        legend.spacing.x=unit(.25,'cm'),
        legend.text=element_text(size=8.7, margin=margin(r=7)),
        plot.margin=margin(5,16,5,25))

activation <- rbind(
  data.frame(t_ms=D$t_ms,category='/ʃ/', p=D$p_sh),
  data.frame(t_ms=D$t_ms,category='/s/',p=D$p_s)
)
activation$category <- factor(activation$category,levels=c('/ʃ/','/s/'))
p_act <- ggplot(activation, aes(t_ms,p,linetype=category,color=category,
                               group=category)) +
  geom_step(linewidth=1.2,direction='hv') + markers() +
  scale_color_manual(values=c('/ʃ/'='#111111','/s/'='#656565'),guide='none') +
  scale_linetype_manual(
    values=c('/ʃ/'='solid','/s/'='longdash'),
    guide=guide_legend(nrow=1,
      override.aes=list(color=c('#111111','#656565'),linewidth=c(1.5,1.5)))) +
  scale_y_continuous(limits=c(0,1),breaks=c(0,.25,.5,.75,1)) +
  x_axis + labs(y='Category support') + base_style +
  theme(legend.position='top',legend.justification='right',
        legend.key.width=unit(1.75,'cm'),
        legend.spacing.x=unit(.35,'cm'),
        axis.title.y=element_text(size=8.0, margin=margin(r=3)),
        plot.margin=margin(3,16,3,25))

same_readiness <- max(abs(D$release_support-D$duration_support),
                      na.rm=TRUE) < 1e-10
if (same_readiness) {
  release_long <- rbind(
    data.frame(t_ms=D$t_ms,source='Duration expectation',
               value=D$duration_support),
    data.frame(t_ms=D$t_ms,source='Following-vowel evidence',
               value=D$vowel_support))
  release_order <- c('Duration expectation',
                     'Following-vowel evidence')
  release_greys <- c('Duration expectation'='#111111',
                     'Following-vowel evidence'='#666666')
  release_styles <- c('Duration expectation'='solid',
                      'Following-vowel evidence'='dotted')
} else {
  release_long <- rbind(
    data.frame(t_ms=D$t_ms,source='Combined readiness',value=D$release_support),
    data.frame(t_ms=D$t_ms,source='Duration expectation',value=D$duration_support),
    data.frame(t_ms=D$t_ms,source='Following-vowel evidence',value=D$vowel_support))
  release_order <- c('Combined readiness','Duration expectation',
                     'Following-vowel evidence')
  release_greys <- c('Combined readiness'='#111111',
                     'Duration expectation'='#555555',
                     'Following-vowel evidence'='#858585')
  release_styles <- c('Combined readiness'='solid',
                      'Duration expectation'='longdash',
                      'Following-vowel evidence'='dotted')
}
release_long$source <- factor(release_long$source,levels=release_order)
p_release <- ggplot(release_long,
                    aes(t_ms,value,linetype=source,color=source,group=source)) +
  geom_step(linewidth=1.1,direction='hv',na.rm=TRUE) + markers() +
  scale_color_manual(values=release_greys,breaks=release_order,guide='none') +
  scale_linetype_manual(values=release_styles,breaks=release_order,
    guide=guide_legend(nrow=1,byrow=TRUE,
      override.aes=list(color=unname(release_greys[release_order]),
                        linewidth=rep(1.5,length(release_order))))) +
  scale_y_continuous(limits=c(0,1),breaks=c(0,.5,1),expand=expansion(mult=c(0,0))) +
  x_axis + labs(y='Boundary evidence') + base_style +
  theme(legend.position='top',legend.justification='center',
        legend.text=element_text(size=9.1,margin=margin(l=3,r=15)),
        legend.key.width=unit(1.65,'cm'),
        legend.spacing.x=unit(.5,'cm'),
        axis.title.y=element_text(size=8.0,margin=margin(r=3)),
        plot.margin=margin(6,16,6,25))

changed <- c(TRUE, D$state[-1] != D$state[-nrow(D)])
starts <- which(changed)
ends <- c(starts[-1]-1,nrow(D))
states <- data.frame(xmin=D$t_ms[starts],
                     xmax=pmin(D$t_ms[ends]+5,XMAX),
                     state=D$state[starts],stringsAsFactors=FALSE)
states$xmid <- (states$xmin+states$xmax)/2
states$kind <- ifelse(grepl('^COMMITTED',states$state),'COMMITTED',states$state)
states$kind <- factor(states$kind,levels=c('IDLE','EVAL','COMMITTED','REPROC'))
states$label <- ifelse(states$xmax-states$xmin>=28,
                       ifelse(states$kind=='COMMITTED','COMMITTED',
                              as.character(states$kind)), '')
state_fills <- c('IDLE'='#f1f1f1','EVAL'='#d3d3d3',
                 'COMMITTED'='#aaaaaa','REPROC'='#6b6b6b')
p_state <- ggplot(states) +
  geom_rect(aes(xmin=xmin,xmax=xmax,ymin=0,ymax=1,fill=kind),
            color='grey35',linewidth=.35) +
  geom_text(aes(x=xmid,y=.5,label=label),family=FONT,size=2.85,
            color='black') +
  geom_segment(data=states[states$kind=='REPROC',,drop=FALSE],
               aes(x=xmid,xend=xmid,y=1.02,yend=1.13),
               inherit.aes=FALSE,linewidth=.45) +
  geom_text(data=states[states$kind=='REPROC',,drop=FALSE],
            aes(x=xmid,y=1.28),label='REPROC',
            inherit.aes=FALSE,family=FONT,size=2.8,color='black') +
  markers() +
  scale_fill_manual(values=state_fills,guide='none') +
  scale_y_continuous(NULL,breaks=NULL,limits=c(0,1.45),expand=c(0,0)) +
  scale_x_continuous('Time (ms)', limits=c(XMIN,XMAX),breaks=XBREAKS,
                     expand=expansion(mult=c(0,0))) +
  theme_classic(base_size=11,base_family=FONT) +
  theme(axis.line.y=element_blank(),axis.ticks.y=element_blank(),
        plot.margin=margin(3,16,4,25))

g1 <- ggplotGrob(p_cues)
g2 <- ggplotGrob(p_release)
g3 <- ggplotGrob(p_act)
g4 <- ggplotGrob(p_state)
shared_width <- unit.pmax(g1$widths,g2$widths,g3$widths,g4$widths)
g1$widths <- shared_width
g2$widths <- shared_width
g3$widths <- shared_width
g4$widths <- shared_width

cairo_pdf(file.path(FIG,'single_trial_revision_trace.pdf'),
          width=9.2,height=6.2,family=FONT)
grid.newpage()
pushViewport(viewport(layout=grid.layout(5,1,
                       heights=unit(c(.22,2.55,1.85,1.55,.98),'null'))))
pushViewport(viewport(layout.pos.row=1))
grid.text('Vowel onset', x=unit(.61,'npc'),y=unit(.45,'npc'),
          gp=gpar(fontfamily=FONT,fontsize=9,col='grey35'))
grid.text('Talker information', x=unit(.82,'npc'),y=unit(.45,'npc'),
          gp=gpar(fontfamily=FONT,fontsize=9,col='grey35'))
popViewport()
for (i in 1:4) {
  pushViewport(viewport(layout.pos.row=i+1))
  grid.draw(list(g1,g2,g3,g4)[[i]])
  popViewport()
}
popViewport()
dev.off()
cat('PASS: wrote figures/single_tracker_control_automaton.pdf\n')
cat('PASS: wrote figures/single_trial_revision_trace.pdf\n')
