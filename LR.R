library(tidyverse)
library(caret)
library(pROC)
library(dplyr)
library(forcats)


clinvar <- read_csv("clinvar_conflicting.csv")

table(train_data$CLASS)
prop.table(table(train_data$CLASS))

#Subset and drop NAs
clinvar_sub <- clinvar %>%
  select(CLASS, AF_ESP, AF_EXAC, AF_TGP, Consequence, IMPACT,
         SIFT, PolyPhen, CADD_PHRED, CADD_RAW, LoFtool, BLOSUM62) %>%
  drop_na()

#Convert relevant columns to factors
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

#Lump rare levels IN THE TRAINING SET ONLY
train_data <- train_data %>%
  mutate(across(all_of(factor_cols), ~ fct_lump_min(.x, min = 10, other_level = "Other")))

#Align factor levels in TEST set to TRAIN set
test_data <- test_data %>%
  mutate(across(all_of(factor_cols), ~ {
    # Keep only levels in train, replace unseen with "Other"
    f <- .
    f[!f %in% levels(train_data[[cur_column()]])] <- "Other"
    factor(f, levels = levels(train_data[[cur_column()]]))
  }))

#Fit logistic regression
logit_model <- glm(
  CLASS ~ AF_ESP + AF_EXAC + AF_TGP + CADD_PHRED + CADD_RAW + LoFtool + BLOSUM62 +
    Consequence + IMPACT + SIFT + PolyPhen,
  data = train_data,
  family = binomial
)

#Predictions
test_probs <- predict(logit_model, newdata = test_data, type = "response")

#Get predicted class labels (choose a threshold, default 0.5)
test_pred_class <- ifelse(test_probs >= 0.5, 1, 0)
test_pred_class <- factor(test_pred_class, levels = c(0,1))

#Confusion matrix
conf_mat <- confusionMatrix(test_pred_class, test_data$CLASS, positive = "1")
print(conf_mat)

#ROC curve and AUC
roc_obj <- roc(test_data$CLASS, test_probs)
plot(roc_obj, main = "ROC Curve", col = "blue")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")
