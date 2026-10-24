args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
  stop("Usage: Rscript lasso.R <input_matrix_path> <output_matrix_path>")
}

if (!requireNamespace("msgps", quietly = TRUE)) {
  stop("The 'msgps' package is required. Install it in R before running lasso.R.")
}

input_matrix_path <- args[[1]]
output_matrix_path <- args[[2]]

expr_matrix <- as.matrix(read.table(input_matrix_path, header = FALSE, sep = "\t"))
storage.mode(expr_matrix) <- "double"

n_samples <- nrow(expr_matrix)
n_genes <- ncol(expr_matrix)

if (n_samples < 2) {
  stop("Need at least 2 samples to fit lasso regression.")
}

if (n_genes < 2) {
  stop("Need at least 2 genes to fit lasso regression.")
}

extract_bic_coefficients <- function(fit) {
  coef_matrix <- coef(fit)

  if (is.null(dim(coef_matrix))) {
    stop("coef(msgps_fit) did not return a matrix as expected.")
  }

  bic_col <- which(tolower(colnames(coef_matrix)) == "bic")
  if (length(bic_col) != 1) {
    stop("Could not find the BIC coefficient column in coef(msgps_fit).")
  }

  bic_coef <- coef_matrix[, bic_col, drop = TRUE]
  names(bic_coef) <- rownames(coef_matrix)
  bic_coef <- bic_coef[names(bic_coef) != "(Intercept)"]
  bic_coef
}

colnames(expr_matrix) <- paste0("G", seq_len(n_genes))
A_lasso <- matrix(0, nrow = n_genes, ncol = n_genes)

for (target_idx in seq_len(n_genes)) {
  predictor_idx <- setdiff(seq_len(n_genes), target_idx)
  X_predictors <- expr_matrix[, predictor_idx, drop = FALSE]
  colnames(X_predictors) <- colnames(expr_matrix)[predictor_idx]
  y_target <- expr_matrix[, target_idx]

  fit <- msgps::msgps(
    X_predictors,
    y_target,
    penalty = "enet",
    alpha = 0,
    intercept = FALSE
  )

  fit_coef <- extract_bic_coefficients(fit)
  A_lasso[target_idx, match(names(fit_coef), colnames(expr_matrix))] <- fit_coef
}

diag(A_lasso) <- 0
write.table(A_lasso, file = output_matrix_path, sep = "\t", row.names = FALSE, col.names = FALSE)
