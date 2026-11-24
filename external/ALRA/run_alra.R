args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript run_alra.R INPUT OUTPUT")
if (!requireNamespace("ALRA", quietly = TRUE)) {
  message("The R package ALRA is required.")
  quit(status = 3L)
}
set.seed(0L)
expression <- as.matrix(read.table(args[1], sep = "\t", header = FALSE, check.names = FALSE))
# The benchmark inputs are already normalized and smoothed.
rank <- ALRA::choose_k(expression)$k
completed <- ALRA::alra(expression, k = rank)[[3]]
write.table(completed, file = args[2], sep = "\t", row.names = FALSE,
            col.names = FALSE, quote = FALSE)
