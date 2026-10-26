#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

FIGURES <- 'figures'
TABLES <- 'tables'

timing_trials_file <- 'demos/output/learning/commitment_timing_trials.csv'
matched_trials_file <- 'demos/output/learning/boundary_locking_audit/matched_duration_trials.csv'
required <- c(timing_trials_file, matched_trials_file)
missing <- required[!file.exists(required)]
if (length(missing)) stop('Missing rerun results: ', paste(missing, collapse = ', '))
manifest_read <- function(path) {
  if (!file.exists(path)) stop('Missing completed-run manifest: ', path,
    '. Regenerate simulations with --reuse-model (no retraining required).')
  d <- read.csv(path, stringsAsFactors = FALSE)
  if (!identical(names(d), c('key', 'value')) || anyDuplicated(d$key))
    stop('Invalid run manifest: ', path)
  setNames(as.character(d$value), d$key)
}
manifest_require <- function(m, keys, path) {
  if (!all(keys %in% names(m))) stop('Incomplete run manifest: ', path)
}
learn_path <- 'demos/output/learning/completed_learning_provenance.csv'
audit_path <- 'demos/output/learning/boundary_locking_audit/completed_audit_provenance.csv'
L <- manifest_read(learn_path)
A <- manifest_read(audit_path)
manifest_require(L, c('controller_md5','model_md5','timing_csv_md5','seed','n_test'), learn_path)
manifest_require(A, c('controller_md5','model_md5','matched_csv_md5','seed','n_test'), audit_path)
for (key in c('controller_md5','model_md5','seed','n_test')) {
  if (L[[key]] != A[[key]]) stop('Mixed controller/model/run provenance (', key, '). Regenerate both simulations.')
}
checks <- c(
  controller_md5 = 'realtime_model/phonetic_competition_model.py',
  model_md5 = 'demos/output/learning/phonetic_full_inventory_model.npz',
  timing_csv_md5 = timing_trials_file
)
for (key in names(checks)) {
  if (unname(tools::md5sum(checks[[key]])) != L[[key]])
    stop('Stale or modified ', checks[[key]], '; regenerate simulations before plotting.')
}
if (unname(tools::md5sum(matched_trials_file)) != A[['matched_csv_md5']])
  stop('Stale matched-duration results; regenerate audit before plotting.')

dir.create(FIGURES, showWarnings = FALSE)
dir.create(TABLES, showWarnings = FALSE)

target_label <- c(
  "d|C_prevocalic"  = "/d/",
  "t|C_prevocalic"  = "/t/",
  "s|C_prevocalic"  = "/s/",
  "sh|C_prevocalic" = "/ʃ/"
)

target_order <- c(
  "/d/",
  "/t/",
  "/s/",
  "/ʃ/"
)

theme_paper <- theme_classic(base_size = 13) +
  theme(
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11.5),
    legend.position = "none",
    plot.margin = margin(8, 12, 8, 8)
  )

# FIGURE 1

trials <- read.csv(
  timing_trials_file,
  stringsAsFactors = FALSE
)

online <- trials[
  trials$field == "full_field" &
  trials$truth %in% names(target_label) &
  !is.na(trials$commit_ms),
]

online$Target <- factor(
  unname(target_label[online$truth]),
  levels = target_order
)

p1 <- ggplot(
  online,
  aes(
    x = Target,
    y = commit_ms,
    fill = Target
  )
) +
  geom_boxplot(
    width = 0.60,
    linewidth = 0.75,
    outlier.shape = NA
  ) +
  geom_jitter(
    position = position_jitter(width = 0.16, height = 0, seed = 42),
    size = 0.75,
    alpha = 0.20,
    color = 'black',
    show.legend = FALSE
  ) +
  scale_fill_manual(values = c("/d/"="#eeeeee", "/t/"="#c9c9c9", "/s/"="#999999", "/ʃ/"="#626262")) +
  scale_y_continuous(
    breaks = pretty(online$commit_ms, n = 8),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = NULL,
    y = "Commitment time from consonant onset (ms)"
  ) +
  theme_paper

ggsave(
  file.path(FIGURES, "realtime_commitment_times.pdf"),
  p1,
  width = 6.3,
  height = 4.4,
  device = cairo_pdf
)

# FIGURE 2
matched <- read.csv(
  matched_trials_file,
  stringsAsFactors = FALSE
)

matched <- matched[
  matched$condition == "full_model" &
  matched$truth == "s|C_prevocalic" &
  !is.na(matched$commit_ms),
]

matched$Duration <- factor(
  matched$duration_shift_ms,
  levels = c(-50, 0, 50),
  labels = c("Short", "Normal", "Long")
)

matched$commit_relative_to_offset_ms <- (
  matched$commit_ms - matched$boundary_ms
)

p2 <- ggplot(
  matched,
  aes(
    x = commit_relative_to_offset_ms,
    y = Duration,
    fill = Duration
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.8,
    color = "black"
  ) +
  geom_boxplot(
    width = 0.55,
    linewidth = 0.75,
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0,
    height = 0.12,
    size = 0.7,
    alpha = 0.3,
    color = "black"
  ) +
  scale_fill_manual(values = c("Short"="#eeeeee", "Normal"="#bdbdbd", "Long"="#777777")) +
  scale_x_continuous(breaks = pretty(matched$commit_relative_to_offset_ms, n = 9)) +
  labs(
    x = "Commitment relative to fricative offset (ms)",
    y = NULL
  ) +
  theme_classic(base_size = 13) +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 13),
    axis.text.x = element_text(size = 11.5),
    axis.text.y = element_text(size = 12, face = "bold"),
    plot.margin = margin(8, 12, 8, 8)
  )

ggsave(
  file.path(FIGURES, "s_commitment_relative_to_offset.pdf"),
  p2,
  width = 7.0,
  height = 3.7,
  device = cairo_pdf
)
