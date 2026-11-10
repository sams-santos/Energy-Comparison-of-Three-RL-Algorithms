# =================================================================
# Statistical analysis of the variable-seed results
# Solution quality (best_distance) and energy until the best solution
# of Q-Learning, SARSA and Double Q-Learning on TSP/ATSP instances
# Last updated: 2026-11-09
# Authors: Marcos Oliveira, Daniela Oliveira
# Project: Energy-Comparison-of-Three-RL-Algorithms
#
# Run from the repository root.
# Inputs: the <instance>_master_summary_best.csv files built by
#   summary_best.py in results/<algorithm>/variable-seed/
# Outputs (tables and figures): results/statistical-analysis/
# =================================================================

# ---------------------------------------------------------
# 1. Packages
# ---------------------------------------------------------

# Install a package if it is missing, then load it
use <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

packages <- c(
  "ARTool",    # Aligned Rank Transform (ART) ANOVA and ART-C contrasts
  "car",       # Levene's test
  "dplyr",
  "flextable", # Formatted tables
  "ggplot2",
  "officer",   # Word documents
  "readr",
  "tidyr"
)

invisible(lapply(packages, use))

# ---------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------

# Input and output directories, relative to the repository root
results_dir <- "results"
seed_mode <- "variable-seed"
output_dir <- file.path(results_dir, "statistical-analysis")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Result folder of each algorithm (same names as in summary_best.py)
algorithm_dirs <- c("sarsa", "q-learning", "double-q")

# Input files (one per instance)
instance_files <- c(
  "br17.atsp_master_summary_best.csv",
  "ftv33.atsp_master_summary_best.csv",
  "eil51.tsp_master_summary_best.csv",
  "berlin52.tsp_master_summary_best.csv",
  "ftv64.atsp_master_summary_best.csv",
  "st70.tsp_master_summary_best.csv",
  "kroA100.tsp_master_summary_best.csv",
  "tsp225.tsp_master_summary_best.csv"
)

# Instance order used in tables and figures
instance_order <- c(
  "eil51.tsp",
  "berlin52.tsp",
  "st70.tsp",
  "kroA100.tsp",
  "tsp225.tsp",
  "br17.atsp",
  "ftv33.atsp",
  "ftv64.atsp"
)

# Conversion factor from kWh (CodeCarbon output) to mWh
kwh_to_mwh <- 1e6

# Marker shape of each reward function in the figures
reward_shapes <- c(R1 = 16, R2 = 15, R3 = 17)

# Correlation between duration and energy (section 10):
# "spearman" (rank-based, robust to non-normality) or "pearson" (linear)
cor_method <- "spearman"
cor_name <- if (cor_method == "spearman") "Spearman" else "Pearson"
cor_symbol <- if (cor_method == "spearman") "rho" else "r"

# ---------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------

# Path of an output file inside output_dir
output_path <- function(file) {
  file.path(output_dir, file)
}

# Helper functions call their package functions with pkg:: so that they
# declare their own dependencies.

# Read the result files of one algorithm and combine them
read_results <- function(directory, files) {
  paths <- file.path(directory, files)
  lapply(paths, readr::read_csv, show_col_types = FALSE) |>
    dplyr::bind_rows()
}

# Export a data frame as a formatted table in a Word document
save_table_docx <- function(data, title, path,
                            font_size = NULL, booktabs = FALSE) {
  ft <- flextable::flextable(data)

  if (booktabs) {
    ft <- ft |>
      flextable::theme_booktabs() |>
      flextable::bold(part = "header") |>
      flextable::align(align = "center", part = "all")
  }

  if (!is.null(font_size)) {
    ft <- flextable::fontsize(ft, size = font_size, part = "all")
  }

  ft <- flextable::autofit(ft)

  doc <- officer::read_docx() |>
    officer::body_add_par(title, style = "heading 1") |>
    flextable::body_add_flextable(value = ft)

  print(doc, target = path)
}

# Save a figure as a high-resolution PNG and as a PDF
save_figure <- function(plot, file_stem, width, height) {
  ggplot2::ggsave(
    filename = paste0(file_stem, ".png"),
    plot = plot,
    width = width, height = height, units = "in", dpi = 300
  )
  ggplot2::ggsave(
    filename = paste0(file_stem, ".pdf"),
    plot = plot,
    width = width, height = height, units = "in"
  )
}

# Convert p-values into significance stars
significance_stars <- function(p) {
  dplyr::case_when(
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ "ns"
  )
}

# Flag the points that no other point dominates (both objectives minimized)
is_nondominated <- function(x, y) {
  vapply(seq_along(x), function(i) {
    !any(x <= x[i] & y <= y[i] & (x < x[i] | y < y[i]))
  }, logical(1))
}

# Format a p-value for the figure labels
format_p <- function(p) {
  ifelse(
    p < 0.001,
    "p < 0.001",
    paste0("p = ", formatC(p, format = "f", digits = 6))
  )
}

# Theme shared by the 2 x 4 panel figures
panel_theme <- theme_bw(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle    = element_text(size = 10, hjust = 0.5),
    strip.text       = element_text(face = "bold", size = 10),
    axis.title       = element_text(face = "bold"),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.spacing    = grid::unit(0.8, "lines")
  )

# ---------------------------------------------------------
# 4. Read the results of all algorithms and set the factors
# ---------------------------------------------------------

all_data <- lapply(algorithm_dirs, function(algorithm_dir) {
  directory <- file.path(results_dir, algorithm_dir, seed_mode)
  read_results(directory, instance_files)
}) |>
  bind_rows() |>
  mutate(
    instance = factor(instance, levels = instance_order),
    algorithm = factor(
      algorithm,
      levels = c("double", "q", "sarsa"),
      labels = c("Double", "Q", "SARSA")
    ),
    r_type = factor(r_type, levels = c("R1", "R2", "R3"))
  )

# =========================================================
# 5. DESCRIPTIVE ANALYSIS OF SOLUTION QUALITY
#    8 instances x 3 algorithms x 3 reward functions x 20 replications
# =========================================================

# ---------------------------------------------------------
# 5.1 Descriptive statistics of best_distance
# ---------------------------------------------------------

table_best_distance <- all_data |>
  group_by(instance, algorithm, r_type) |>
  summarise(
    optimal_distance = first(optimal_distance),
    min    = min(best_distance, na.rm = TRUE),
    max    = max(best_distance, na.rm = TRUE),
    median = median(best_distance, na.rm = TRUE),
    mean   = mean(best_distance, na.rm = TRUE),
    sd     = sd(best_distance, na.rm = TRUE),
    IQR    = IQR(best_distance, na.rm = TRUE),
    .groups = "drop"
  )

save_table_docx(
  table_best_distance,
  title = "Descriptive Statistics of Best Distance",
  path = output_path("10 - Descriptive Statistics Best Distance.docx"),
  font_size = 8,
  booktabs = TRUE
)

# ---------------------------------------------------------
# 5.2 Minimum distances (following Table 2 of Wang, 2023)
# ---------------------------------------------------------

# Known optimal distance of each instance (TSPLIB values recorded in the
# optimal_distance column of the input files)
optimal_values <- all_data |>
  distinct(instance, best_distance = optimal_distance) |>
  arrange(instance)

# Minimum distance across replications, per algorithm and instance
min_distance <- all_data |>
  group_by(algorithm, instance) |>
  summarise(best_distance = min(best_distance), .groups = "drop") |>
  pivot_wider(names_from = instance, values_from = best_distance)

# Row with the optimal values
optimal_row <- optimal_values |>
  pivot_wider(names_from = instance, values_from = best_distance) |>
  mutate(algorithm = "Optimal Solution") |>
  select(algorithm, everything())

# Combine the rows in the desired order
table_min_distance <- bind_rows(
  min_distance |> filter(algorithm == "Double"),
  min_distance |> filter(algorithm == "Q"),
  min_distance |> filter(algorithm == "SARSA"),
  optimal_row
) |>
  select(algorithm, all_of(instance_order))

save_table_docx(
  table_min_distance,
  title = "Minimum Best Distance by Algorithm and Instance",
  path = output_path("11 - Minimum Distances Table.docx")
)

# ---------------------------------------------------------
# 5.3 Boxplots of best_distance
#     8 instances in a 2 x 4 grid; 3 algorithms x 3 reward functions
# ---------------------------------------------------------

figure_boxplot <- ggplot(
  all_data,
  aes(x = algorithm, y = best_distance, fill = r_type)
) +
  geom_boxplot(
    position = position_dodge(width = 0.8),
    width = 0.65,
    outlier.size = 0.8,
    outlier.alpha = 0.6
  ) +
  # One panel per instance
  facet_wrap(~instance, ncol = 4, scales = "free_y") +
  labs(
    title    = "Best Distances by Algorithm and Reward Function",
    subtitle = paste(
      "Distribution across 20 replications |",
      "3 algorithms x 3 reward functions"
    ),
    x        = "Algorithm",
    y        = "Best Distance",
    fill     = "Reward Function"
  ) +
  panel_theme +
  theme(axis.text.x = element_text(size = 9))

print(figure_boxplot)

save_figure(
  figure_boxplot,
  output_path("12 - Boxplot Best Distance"),
  width = 16, height = 9
)

# =========================================================
# 6. 3 x 3 FACTORIAL ANOVA PER INSTANCE
#    Step 1: assumption checks
#    (Shapiro-Wilk on the residuals and Levene's test)
#    Dependent variable: best_distance
# =========================================================

table_assumptions <- all_data |>
  group_by(instance) |>
  group_modify(~ {
    model <- lm(best_distance ~ algorithm * r_type, data = .x)
    levene <- leveneTest(
      best_distance ~ algorithm * r_type,
      data = .x,
      center = median
    )

    data.frame(
      n         = nrow(.x),
      p_Shapiro = shapiro.test(residuals(model))$p.value,
      p_Levene  = levene[1, "Pr(>F)"]
    )
  }) |>
  ungroup() |>
  mutate(
    across(starts_with("p_"), ~ round(.x, 4)),
    normality        = if_else(p_Shapiro > 0.05, "Not rejected", "Rejected"),
    homoscedasticity = if_else(p_Levene > 0.05, "Not rejected", "Rejected")
  )

print(table_assumptions)

save_table_docx(
  table_assumptions,
  title = paste(
    "Normality and Homoscedasticity Assumption Checks by Instance.",
    "DV: Best Distance"
  ),
  path = output_path("13 - Assumption Checks.docx"),
  font_size = 8
)

# ---------------------------------------------------------
# The assumption checks (file 13) rejected homoscedasticity in
# every instance and normality in most of them. Therefore, the
# non-parametric factorial ART (Aligned Rank Transform) ANOVA
# was used instead.
# ---------------------------------------------------------

# =========================================================
# 7. FACTORIAL ART ANOVA PER INSTANCE
#    Dependent variable: best_distance
#    Factors: algorithm x r_type
# =========================================================

table_art <- all_data |>
  group_by(instance) |>
  group_modify(~ {
    art(best_distance ~ algorithm * r_type, data = .x) |>
      anova() |>
      as.data.frame() |>
      transmute(
        effect  = Term,
        df      = Df,
        df_res  = Df.res,
        F_ART   = `F value`,
        p_value = `Pr(>F)`
      )
  }) |>
  ungroup() |>
  # Holm correction across instances, separately for each effect
  group_by(effect) |>
  mutate(p_Holm = p.adjust(p_value, method = "holm")) |>
  ungroup() |>
  mutate(
    # car also exports recode(), so the dplyr version is called explicitly
    effect = dplyr::recode(
      effect,
      "algorithm"        = "Algorithm",
      "r_type"           = "Reward",
      "algorithm:r_type" = "Interaction"
    ),
    significance = significance_stars(p_Holm),
    F_ART   = round(F_ART, 4),
    p_value = round(signif(p_value, 4), 4),
    p_Holm  = round(signif(p_Holm, 4), 4)
  )

print(table_art, n = Inf)

save_table_docx(
  table_art,
  title = paste(
    "Results of the Non-Parametric Factorial ART ANOVA.",
    "DV: Best Distance"
  ),
  path = output_path("14 - Factorial ART Results.docx"),
  font_size = 8
)

# =========================================================
# 8. ART-C MULTIPLE COMPARISONS: 9 algorithm x r_type combinations
#    Per instance, Holm adjustment (36 comparisons per instance)
# =========================================================

comparisons_art <- all_data |>
  group_by(instance) |>
  group_modify(~ {
    model <- art(best_distance ~ algorithm * r_type, data = .x)
    as.data.frame(art.con(model, "algorithm:r_type", adjust = "holm"))
  }) |>
  ungroup() |>
  mutate(
    significance = significance_stars(p.value),
    estimate = round(estimate, 4),
    t.ratio  = round(t.ratio, 4),
    p.value  = round(signif(p.value, 4), 4)
  )

print(comparisons_art, n = Inf)

save_table_docx(
  comparisons_art,
  title = paste(
    "Multiple Comparisons from the Non-Parametric Factorial ART ANOVA.",
    "DV: Best Distance"
  ),
  path = output_path("15 - Factorial ART Multiple Comparisons.docx"),
  font_size = 8
)

# =========================================================
# 9. PARETO ANALYSIS: best_distance x Energy_until_Best_kWh
#    One panel per instance (2 x 4 grid)
#    9 points per panel (3 algorithms x 3 reward functions)
# =========================================================

# ---------------------------------------------------------
# 9.1 Means per instance x algorithm x r_type (energy in mWh) and
#     non-dominated points (minimization of both objectives)
# ---------------------------------------------------------

pareto_data <- all_data |>
  group_by(instance, algorithm, r_type) |>
  summarise(
    mean_distance = mean(best_distance, na.rm = TRUE),
    mean_energy   = mean(Energy_until_Best_kWh, na.rm = TRUE) * kwh_to_mwh,
    .groups = "drop"
  ) |>
  group_by(instance) |>
  mutate(pareto = is_nondominated(mean_distance, mean_energy)) |>
  ungroup()

# Front of each instance (sorted by x, used to draw the staircase)
pareto_front <- pareto_data |>
  filter(pareto) |>
  arrange(instance, mean_distance)

# ---------------------------------------------------------
# 9.2 Table of the non-dominated combinations
# ---------------------------------------------------------

table_pareto <- pareto_front |>
  select(instance, algorithm, r_type, mean_distance, mean_energy) |>
  mutate(
    mean_distance = round(mean_distance, 2),
    mean_energy   = round(mean_energy, 2)
  ) |>
  rename(mean_energy_mWh = mean_energy)

print(table_pareto, n = Inf)

save_table_docx(
  table_pareto,
  title = "Non-Dominated Combinations (Pareto Front) by Instance",
  path = output_path("16 - Pareto Front Table.docx"),
  font_size = 8,
  booktabs = TRUE
)

# ---------------------------------------------------------
# 9.3 Pareto figure: 2 x 4 grid
# ---------------------------------------------------------

figure_pareto <- ggplot(pareto_data, aes(x = mean_distance, y = mean_energy)) +
  # Front (staircase), drawn only where it has more than one point
  geom_step(
    data = pareto_front |> filter(n() > 1, .by = instance),
    direction = "hv",
    colour = "grey40", linetype = "dashed", linewidth = 0.5
  ) +
  # Ring around the non-dominated points
  geom_point(
    data = pareto_front,
    shape = 1, size = 6, stroke = 0.9, colour = "red"
  ) +
  # The 9 points: colour = algorithm, shape = reward function
  geom_point(aes(colour = algorithm, shape = r_type), size = 3.5) +
  facet_wrap(~instance, nrow = 2, ncol = 4, scales = "free") +
  scale_shape_manual(values = reward_shapes) +
  labs(
    title    = "Pareto Analysis: Solution Quality vs Energy Consumption",
    subtitle = paste(
      "Means over 20 replications |",
      "Red rings and dashed line: non-dominated combinations (minimization)"
    ),
    x        = "Mean Best Distance",
    y        = "Mean Energy until Best (mWh)",
    colour   = "Algorithm",
    shape    = "Reward Function"
  ) +
  panel_theme

print(figure_pareto)

save_figure(
  figure_pareto,
  output_path("17 - Pareto Best Distance vs Energy"),
  width = 16, height = 9
)

# =========================================================
# 10. SCATTER PLOTS: Duration_until_Best_sec vs Energy_until_Best_kWh
#     One panel per instance (2 x 4 grid)
#     Correlation coefficient and p-value in the upper-left corner
# =========================================================

# ---------------------------------------------------------
# 10.1 Correlation per instance (method set by cor_method)
# ---------------------------------------------------------

table_cor <- all_data |>
  group_by(instance) |>
  group_modify(~ {
    test <- cor.test(
      .x$Duration_until_Best_sec,
      .x$Energy_until_Best_kWh * kwh_to_mwh,
      method = cor_method,
      exact = FALSE # Avoids warnings with ties (ignored by Pearson)
    )
    data.frame(
      n       = nrow(.x),
      r       = unname(test$estimate),
      p_value = test$p.value
    )
  }) |>
  ungroup() |>
  mutate(
    label = paste0(
      cor_symbol, " = ",
      formatC(r, format = "f", digits = 6),
      "\n",
      format_p(p_value)
    )
  )

print(table_cor, n = Inf)

# ---------------------------------------------------------
# 10.2 Scatter figure: 2 x 4 grid
# ---------------------------------------------------------

figure_scatter <- ggplot(
  all_data,
  aes(x = Duration_until_Best_sec, y = Energy_until_Best_kWh * kwh_to_mwh)
) +
  geom_point(aes(colour = algorithm, shape = r_type), size = 1.8, alpha = 0.7) +
  geom_smooth(
    method = "lm", formula = y ~ x, se = FALSE,
    colour = "black", linewidth = 0.5, linetype = "dashed"
  ) +
  # Correlation label in the upper-left corner of each panel
  geom_text(
    data = table_cor,
    aes(x = -Inf, y = Inf, label = label),
    hjust = -0.08, vjust = 1.3, size = 3.4, fontface = "bold",
    inherit.aes = FALSE
  ) +
  facet_wrap(~instance, nrow = 2, ncol = 4, scales = "free") +
  scale_shape_manual(values = reward_shapes) +
  labs(
    title    = "Duration vs Energy until the Best Solution",
    subtitle = paste0(
      "Each point is one replication | Dashed line: linear fit | ",
      cor_name, " correlation"
    ),
    x        = "Duration until Best (s)",
    y        = "Energy until Best (mWh)",
    colour   = "Algorithm",
    shape    = "Reward Function"
  ) +
  panel_theme

print(figure_scatter)

save_figure(
  figure_scatter,
  output_path("18 - Scatter Duration vs Energy"),
  width = 16, height = 9
)

# ---------------------------------------------------------
# 10.3 Correlation table in Word
# ---------------------------------------------------------

table_cor_docx <- table_cor |>
  transmute(
    Instance     = as.character(instance),
    n            = n,
    coefficient  = formatC(r, format = "f", digits = 4),
    # p-values this small are below numerical precision: report p < 0.001
    `p-value`    = if_else(
      p_value < 0.001,
      "< 0.001",
      formatC(p_value, format = "f", digits = 3)
    ),
    Significance = significance_stars(p_value)
  ) |>
  rename(!!paste0(cor_name, "'s ", cor_symbol) := coefficient)

save_table_docx(
  table_cor_docx,
  title = paste(
    cor_name,
    "Correlation between Duration and Energy until Best by Instance"
  ),
  path = output_path("19 - Correlation Duration Energy.docx"),
  font_size = 9,
  booktabs = TRUE
)

# =========================================================
# 11. SCATTER PLOTS: Episodes until Best (%) vs Energy until Best (%)
#     One panel per instance (2 x 4 grid)
#     Dashed red line: y = x (energy spent in proportion to episodes)
# =========================================================

# ---------------------------------------------------------
# 11.1 Percentage variables
# ---------------------------------------------------------

# best_episode is 0-based and the energy until the best covers
# episodes 0..best_episode, i.e. best_episode + 1 episodes
all_data <- all_data |>
  mutate(
    energy_until_pct   = 100 * Energy_until_Best_kWh / Total_Energy_kWh,
    episodes_until_pct = 100 * (best_episode + 1) / episodes
  )

# ---------------------------------------------------------
# 11.2 Data for the figure
#      (instance is already a factor ordered by instance_order)
# ---------------------------------------------------------

data_episodes <- all_data |>
  filter(
    !is.na(episodes_until_pct),
    !is.na(energy_until_pct),
    !is.na(instance)
  )

# Common axis limit for all panels
axis_max <- max(
  data_episodes$episodes_until_pct,
  data_episodes$energy_until_pct,
  na.rm = TRUE
)

# ---------------------------------------------------------
# 11.3 Scatter figure: 2 x 4 grid
# ---------------------------------------------------------

figure_episodes <- ggplot(
  data_episodes,
  aes(x = episodes_until_pct, y = energy_until_pct)
) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(
    intercept = 0, slope = 1,
    linetype = "dashed", colour = "red", linewidth = 0.7
  ) +
  facet_wrap(~instance, nrow = 2, ncol = 4, scales = "fixed") +
  coord_equal(xlim = c(0, axis_max), ylim = c(0, axis_max)) +
  labs(
    x = "Episodes until Best (%)",
    y = "Energy until Best (%)"
  ) +
  theme_bw(base_size = 10) +
  theme(
    strip.text    = element_text(face = "bold", size = 9),
    axis.title    = element_text(size = 10),
    axis.text     = element_text(size = 8),
    panel.spacing = grid::unit(0.8, "lines")
  )

print(figure_episodes)

save_figure(
  figure_episodes,
  output_path("20 - Episodes vs Energy 8 Instances"),
  width = 12, height = 6.5
)
