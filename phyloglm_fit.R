# Phylogenetic logistic regression (phylolm::phyloglm) on the datasets
# exported by phyloglm_fit.py. Not meant to be run by hand: phyloglm_fit.py
# writes the inputs, calls this script and collects the outputs.
#
# Usage: Rscript phyloglm_fit.R <work_dir> <method> <btol> <log_alpha_bound> <boot>
#

suppressPackageStartupMessages({
    library(ape)
    library(phylolm)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    stop("Usage: Rscript phyloglm_fit.R <work_dir> <method> <btol> <log_alpha_bound> <boot>")
}
work_dir        <- args[1]
method          <- args[2]
btol            <- as.numeric(args[3])
log_alpha_bound <- as.numeric(args[4])
boot            <- as.integer(args[5])

manifest <- read.csv(file.path(work_dir, "manifest.csv"), stringsAsFactors = FALSE)
cat("phylolm", as.character(packageVersion("phylolm")), "| method:", method,
    "| btol:", btol, "| log.alpha.bound:", log_alpha_bound, "| boot:", boot,
    "| datasets:", nrow(manifest), "\n\n")

params <- vector("list", nrow(manifest))

for (i in seq_len(nrow(manifest))) {
    k <- manifest$k[i]
    train <- read.csv(file.path(work_dir, manifest$train[i]), stringsAsFactors = FALSE)
    test  <- read.csv(file.path(work_dir, manifest$test[i]),  stringsAsFactors = FALSE)
    tre   <- read.tree(file.path(work_dir, manifest$tree[i]))

    # phyloglm matches data rows to tips by row name
    rownames(train) <- train$tip
    x_cols <- setdiff(colnames(train), c("tip", "y"))
    dat <- train[, c("y", x_cols)]
    fml <- as.formula(paste("y ~", paste(x_cols, collapse = " + ")))

    start <- Sys.time()
    status <- "ok"; msg <- ""
    fit <- withCallingHandlers(
        tryCatch({
            # the model is fitted on the training species only
            tre_train <- keep.tip(tre, train$tip)
            phyloglm(fml, data = dat, phy = tre_train, method = method,
                     btol = btol, log.alpha.bound = log_alpha_bound, boot = boot)
            },
            error = function(e) { status <<- "error"; msg <<- conditionMessage(e); NULL }
        ),
        warning = function(w) {
            msg <<- paste(c(msg[msg != ""], conditionMessage(w)), collapse = " | ")
            invokeRestart("muffleWarning")
        }
    )
    elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))

    row <- list(k = k, status = status, message = gsub("[\r\n,]", " ", msg),
                alpha = NA, convergence = NA, alphaWarn = NA,
                logLik = NA, aic = NA, n_train = nrow(train), time_sec = elapsed)

    if (is.null(fit)) {
        prob <- rep(NA_real_, nrow(test))
    } else {
        # no predict() method for phyloglm: probabilities from the fitted
        # coefficients, P(y = 1 | x) = plogis(x' beta)
        beta <- coef(fit)
        Xt <- model.matrix(as.formula(paste("~", paste(x_cols, collapse = " + "))), test)
        prob <- as.vector(plogis(Xt[, names(beta), drop = FALSE] %*% beta))

        if (fit$convergence != 0) row$status <- "no_convergence"
        row$alpha <- fit$alpha
        row$convergence <- fit$convergence
        row$alphaWarn <- fit$alphaWarn
        row$logLik <- fit$logLik
        row$aic <- fit$aic
        for (nm in names(beta)) row[[paste0("coef_", make.names(nm))]] <- unname(beta[nm])
    }

    write.csv(data.frame(prob = prob), file.path(work_dir, paste0("pred_", k, ".csv")),
              row.names = FALSE)
    params[[i]] <- row

    cat(sprintf("[%d/%d] k=%s status=%s alpha=%s time=%.1fs %s\n", i, nrow(manifest),
                k, status, format(row$alpha, digits = 4), elapsed,
                if (msg != "") paste("|", msg) else ""))
}

# rows of failed fits have no coef_ columns: fill them with NA
all_cols <- unique(unlist(lapply(params, names)))
params_df <- do.call(rbind, lapply(params, function(r) {
    r[setdiff(all_cols, names(r))] <- NA
    as.data.frame(r[all_cols], stringsAsFactors = FALSE)
}))
write.csv(params_df, file.path(work_dir, "params.csv"), row.names = FALSE)
cat("\nDone:", sum(params_df$status != "error"), "of", nrow(params_df), "fits ran,",
    sum(params_df$status == "no_convergence"), "did not converge,",
    sum(params_df$status == "error"), "failed\n")
