args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript lm.R <input_matrix_path> <output_matrix_path>")
}

input_matrix_path <- args[[1]]
output_matrix_path <- args[[2]]

expr_matrix <- as.matrix(read.table(input_matrix_path, header = FALSE, sep = "\t"))
storage.mode(expr_matrix) <- "double"

n_samples <- nrow(expr_matrix)
n_genes <- ncol(expr_matrix)

if (n_samples < 2) {
  stop("Need at least 2 samples to fit linear regression.")
}

if (n_genes < 2) {
  stop("Need at least 2 genes to fit linear regression.")
}

colnames(expr_matrix) <- paste0("G", seq_len(n_genes))
A_lm <- matrix(0, nrow = n_genes, ncol = n_genes)

for (target_idx in seq_len(n_genes)) {
  response_name <- colnames(expr_matrix)[target_idx]
  predictor_names <- colnames(expr_matrix)[-target_idx]

  fit_data <- as.data.frame(expr_matrix[, c(target_idx, setdiff(seq_len(n_genes), target_idx)), drop = FALSE])
  colnames(fit_data) <- c(response_name, predictor_names)

  fit <- lm(reformulate(predictor_names, response = response_name, intercept = FALSE), data = fit_data)
  fit_coef <- coef(fit)

  A_lm[target_idx, match(names(fit_coef), colnames(expr_matrix))] <- fit_coef
}

diag(A_lm) <- 0
write.table(A_lm, file = output_matrix_path, sep = "\t", row.names = FALSE, col.names = FALSE)
