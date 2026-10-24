suppressPackageStartupMessages({
  library(infotheo)
  library(pcalg)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript run_mi_ida_data1.R <input_matrix> <output_dir> <tfnum> <cnum> [alpha]")
}

input_matrix <- args[[1]]
output_dir <- args[[2]]
tfnum <- as.integer(args[[3]])
cnum <- as.integer(args[[4]])
alpha <- if (length(args) >= 5) as.numeric(args[[5]]) else 0.01

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

raw_mat <- as.matrix(read.table(input_matrix, sep = "\t"))
if (nrow(raw_mat) < tfnum || ncol(raw_mat) < cnum) {
  stop(sprintf(
    "Input matrix shape (%d x %d) is smaller than requested tfnum=%d, cnum=%d",
    nrow(raw_mat), ncol(raw_mat), tfnum, cnum
  ))
}

X <- t(raw_mat[1:tfnum, 1:cnum, drop = FALSE])
Xdisc <- discretize(X)
A_MI <- mutinformation(Xdisc, method = "mm")
write.table(
  A_MI,
  file = file.path(output_dir, "A_MI.txt"),
  row.names = FALSE,
  col.names = FALSE,
  sep = "\t"
)

suffStat <- list(C = cor(X), n = cnum)
pc.fit <- pc(suffStat, indepTest = gaussCItest, p = tfnum, alpha = alpha)
covX <- cov(X)

PredictedA_min <- matrix(0, tfnum, tfnum)
PredictedA_mean <- matrix(0, tfnum, tfnum)

for (i in seq_len(tfnum)) {
  temp <- abs(idaFast(i, seq_len(tfnum), covX, pc.fit@graph))
  PredictedA_min[, i] <- apply(temp, 1, FUN = min)
  PredictedA_mean[, i] <- rowMeans(temp)
}

write.table(
  PredictedA_min,
  file = file.path(output_dir, "A_IDA_min.txt"),
  row.names = FALSE,
  col.names = FALSE,
  sep = "\t"
)

write.table(
  PredictedA_mean,
  file = file.path(output_dir, "A_IDA_mean.txt"),
  row.names = FALSE,
  col.names = FALSE,
  sep = "\t"
)
