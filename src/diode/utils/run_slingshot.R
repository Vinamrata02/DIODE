args <- commandArgs(trailingOnly = TRUE)
expr_path <- args[1]
out_path <- args[2]
k_arg <- suppressWarnings(as.integer(args[3]))
start_cell <- suppressWarnings(as.integer(args[4]))
seed <- suppressWarnings(as.integer(args[5]))
dist_method <- args[6]
if (is.na(seed)) seed <- 1
if (is.na(dist_method) || dist_method == "") dist_method <- "slingshot"
suppressPackageStartupMessages(library(slingshot))

expr <- as.matrix(read.csv(expr_path, row.names = 1, check.names = FALSE))
embedding <- t(expr)
n_cells <- nrow(embedding)
if (is.na(start_cell) || start_cell < 1 || start_cell > n_cells) start_cell <- 1

if (!all(is.finite(embedding))) {
    stop("Input embedding contains non-finite values.")
}

unique_rows <- nrow(unique(round(embedding, 10)))
if (unique_rows < 3 || n_cells < 3) {
    write.csv(data.frame(cell_id = colnames(expr), pseudotime = rep(0, n_cells)),
              out_path, row.names = FALSE)
    quit(save = "no", status = 0)
}

if (is.na(k_arg)) {
    k_arg <- max(3, min(8, round(sqrt(n_cells / 20))))
}
k <- max(2, min(k_arg, unique_rows, n_cells - 1))

set.seed(seed)
km <- stats::kmeans(embedding, centers = k, nstart = 25, iter.max = 100)
cluster_labels <- paste0("C", km$cluster)
start_cluster <- cluster_labels[start_cell]

sds <- slingshot::slingshot(
    embedding,
    clusterLabels = cluster_labels,
    start.clus = start_cluster,
    dist.method = dist_method
)
pseudotime_matrix <- as.matrix(slingshot::slingPseudotime(sds))
coverage <- colSums(is.finite(pseudotime_matrix))
if (length(coverage) == 0 || max(coverage) == 0) {
    stop("Slingshot returned no finite pseudotime values.")
}

best_lineage <- which.max(coverage)
pseudotime_values <- pseudotime_matrix[, best_lineage]
if (any(!is.finite(pseudotime_values))) {
    cluster_medians <- tapply(pseudotime_values, cluster_labels, stats::median, na.rm = TRUE)
    finite_median <- stats::median(pseudotime_values[is.finite(pseudotime_values)], na.rm = TRUE)
    missing <- which(!is.finite(pseudotime_values))
    for (idx in missing) {
        fill <- cluster_medians[[cluster_labels[idx]]]
        if (!is.finite(fill)) fill <- finite_median
        pseudotime_values[idx] <- fill
    }
}
if (!all(is.finite(pseudotime_values))) {
    stop("Slingshot pseudotime still contains non-finite values after filling missing values.")
}

write.csv(data.frame(cell_id = colnames(expr),
                     pseudotime = pseudotime_values,
                     cluster = cluster_labels,
                     lineage = best_lineage,
                     n_lineages = ncol(pseudotime_matrix),
                     n_clusters = k,
                     start_cluster = start_cluster,
                     dist_method = dist_method),
          out_path, row.names = FALSE)
