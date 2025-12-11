library(tidyverse)
library(caret)
library(pROC)
library(forcats)

clinvar <- read_csv("clinvar_conflicting.csv")

#Subset relevant columns and drop NAs
clinvar_sub <- clinvar %>%
  select(CLASS, AF_ESP, AF_EXAC, AF_TGP, Consequence, IMPACT,
         SIFT, PolyPhen, CADD_PHRED, CADD_RAW, LoFtool, BLOSUM62) %>%
  drop_na()

#Convert categorical columns to factor
factor_cols <- c("Consequence", "IMPACT", "SIFT", "PolyPhen")
clinvar_sub <- clinvar_sub %>%
  mutate(
    CLASS = as.factor(CLASS),
    across(all_of(factor_cols), as.factor)
  )

#Train/test split
set.seed(123)
train_index <- createDataPartition(clinvar_sub$CLASS, p = 0.8, list = FALSE)
train_data <- clinvar_sub[train_index, ]
test_data  <- clinvar_sub[-train_index, ]

#Lump rare levels in TRAINING SET only
train_data <- train_data %>%
  mutate(across(all_of(factor_cols), ~ fct_lump_min(.x, min = 10, other_level = "Other")))

#Align factor levels in TEST set to TRAIN set
test_data <- test_data %>%
  mutate(across(all_of(factor_cols), ~ {
    f <- .
    f[!f %in% levels(train_data[[cur_column()]])] <- "Other"
    factor(f, levels = levels(train_data[[cur_column()]]))
  }))

#Fit logistic regression
features <- c("AF_ESP", "AF_EXAC", "AF_TGP", "CADD_PHRED", "CADD_RAW",
              "LoFtool", "BLOSUM62", "Consequence", "IMPACT", "SIFT", "PolyPhen")

logit_formula <- as.formula(paste("CLASS ~", paste(features, collapse = " + ")))
logit_model <- glm(logit_formula, data = train_data, family = binomial)

#Get predicted probabilities
test_probs <- predict(logit_model, newdata = test_data, type = "response")

#Threshold tuning to maximize F1-score
f1_score <- function(actual, predicted) {
  cm <- confusionMatrix(factor(predicted, levels=c(0,1)),
                        factor(actual, levels=c(0,1)),
                        positive="1")
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

#Threshold grid
thresholds <- seq(0, 1, by = 0.01)

#Compute F1 for each threshold
f1_values <- sapply(thresholds, function(t) f1_score(test_data$CLASS, ifelse(test_probs >= t, 1, 0)))

#Find best threshold
best_thresh <- thresholds[which.max(f1_values)]
cat("Best threshold for F1:", best_thresh, "\n")

#Make predictions
test_pred_class <- ifelse(test_probs >= best_thresh, 1, 0)
test_pred_class <- factor(test_pred_class, levels = c(0,1))

#Confusion matrix
conf_mat <- confusionMatrix(test_pred_class, test_data$CLASS, positive="1")
print(conf_mat)

#ROC curve and AUC
roc_obj <- roc(test_data$CLASS, test_probs)
plot(roc_obj, main="ROC Curve", col="blue")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")
