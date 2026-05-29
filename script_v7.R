# Script version 7
# Last updated: 27/03/2026
# Authors: Marcos Oliveira, Daniela Oliveira
# Project: Energy-Comparison-of-Three-RL-Algorithms

# Function to automatically install and load packages
use <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Install/load the required packages
use("car")
use("dplyr")
use("ScottKnott")
use("flextable")
use("officer")
use("ggplot2")
use("ggpubr")

# Set the results directory
setwd("results")

# Read the data
dados_DQ <- read.csv(
  file.path("double-q", "master_summary.csv"),
  header = TRUE,
  sep = ",",
  quote = "\"",
  stringsAsFactors = FALSE
)

dados_Q <- read.csv(
  file.path("q-learning", "master_summary.csv"),
  header = TRUE,
  sep = ",",
  quote = "\"",
  stringsAsFactors = FALSE
)

dados_S <- read.csv(
  file.path("sarsa", "master_summary.csv"),
  header = TRUE,
  sep = ",",
  quote = "\"",
  stringsAsFactors = FALSE
)

# Filter the data for the fixed epsilon strategy with epsilon = 0.01
dados_reduzido_DQ <- subset(dados_DQ, e_type == "fixed" & epsilon == 0.01)
dados_reduzido_Q  <- subset(dados_Q,  e_type == "fixed" & epsilon == 0.01)
dados_reduzido_S  <- subset(dados_S,  e_type == "fixed" & epsilon == 0.01)

# Combine all datasets into one data frame
dados_todos <- rbind(dados_reduzido_DQ, dados_reduzido_Q, dados_reduzido_S)

# Redefine the order of the instances
dados_todos$instance <- factor(
  dados_todos$instance,
  levels = c(
    "br17.atsp", "ftv33.atsp", "eil51.tsp", "berlin52.tsp",
    "ftv64.atsp", "st70.tsp", "kroA100.tsp", "tsp225.tsp"
  )
)

# Redefine the order of the algorithms
dados_todos$algorithm <- factor(
  dados_todos$algorithm,
  levels = c("q", "sarsa", "double")
)

# Calculate the mean and standard deviation of Total Energy
tabela_medias_total_energy <- dados_todos %>%
  group_by(instance, algorithm, r_type) %>%
  summarise(
    mean = mean(Total_Energy_Wh, na.rm = TRUE),
    sd   = sd(Total_Energy_Wh, na.rm = TRUE),
    n    = n(),
    .groups = "drop"
  )

# Export the summary table to a Word document
ft <- flextable(tabela_medias_total_energy)
doc <- read_docx()
doc <- body_add_flextable(doc, ft)
print(doc, target = "10 - Mean_Total_Energy_Table.docx")

# Bar plot of mean Total Energy by algorithm and reward function
ggplot(
  dados_todos,
  aes(
    x = algorithm,
    y = Total_Energy_Wh,
    fill = r_type
  )
) +
  stat_summary(
    fun = mean,
    geom = "bar",
    position = position_dodge(width = 0.8)
  ) +
  facet_wrap(~ instance, nrow = 3, ncol = 3, scales = "free_y") +
  theme_minimal(base_size = 14) +
  labs(
    x = "Algorithm",
    y = "Mean Total Energy (Wh)",
    fill = "Reward function"
  ) +
  theme(
    legend.position = "top"
  )

# Convert variables to factors
dados_todos$algorithm <- factor(dados_todos$algorithm)
dados_todos$r_type    <- factor(dados_todos$r_type)
dados_todos$instance  <- factor(dados_todos$instance)

# Repeated 3 × 3 factorial ANOVA for all instances
lista_resultados <- list()

dados_todos$trat <- interaction(
  dados_todos$algorithm,
  dados_todos$r_type,
  drop = TRUE
)

# Factorial analysis using Total Energy as the dependent variable
sink("11 - Total_Energy_Wh_ANOVA_Results.txt")

for (inst in levels(dados_todos$instance)) {
  
  cat("\n=====================================================\n")
  cat("FACTORIAL ANOVA — INSTANCE:", inst, "\n")
  cat("=====================================================\n\n")
  
  dados_inst <- subset(dados_todos, instance == inst)
  
  modelo <- aov(Total_Energy_Wh ~ algorithm * r_type, data = dados_inst)
  res <- residuals(modelo)
  
  cat("\n--- ANOVA Table ---\n")
  print(summary(modelo))
  
  cat("\n--- Shapiro-Wilk Test ---\n")
  print(shapiro.test(res))
  
  cat("\n--- Levene's Test ---\n")
  print(leveneTest(Total_Energy_Wh ~ algorithm * r_type, data = dados_inst))
  
  cat("\n--- Scott-Knott Test (algorithm:r_type) ---\n")
  modelo_trat <- aov(Total_Energy_Wh ~ trat, data = dados_inst)
  sk <- SK(modelo_trat)
  print(sk)
}

sink()

# Bar plot of mean Total Energy by algorithm and instance
ggplot(
  dados_todos,
  aes(
    x = algorithm,
    y = Total_Energy_Wh,
    fill = instance
  )
) +
  stat_summary(
    fun = mean,
    geom = "bar",
    position = position_dodge(width = 0.8)
  ) +
  theme_minimal(base_size = 14) +
  labs(
    x = "Algorithm",
    y = "Mean Total Energy (Wh)",
    fill = "Instance"
  ) +
  theme(
    legend.position = "top"
  )

# Plot of Duration versus Total Energy for each instance
ggplot(
  dados_todos,
  aes(
    x = Duration_sec,
    y = Total_Energy_Wh
  )
) +
  geom_point() +
  stat_cor(
    method = "pearson",
    label.x.npc = "left",
    label.y.npc = "top",
    size = 4
  ) +
  facet_wrap(~ instance, scales = "free") +
  labs(
    x = "Duration (sec)",
    y = "Total Energy (Wh)"
  ) +
  theme_bw()