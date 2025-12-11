library(tidyverse)
library(caret)
library(xgboost)
library(fastDummies)
library(pROC)

clinvar_sub <- clinvar %>%
  select(CLASS, AF_ESP, AF_EXAC, AF_TGP, Consequence, IMPACT,
         SIFT, PolyPhen, CADD_PHRED, CADD_RAW, LoFtool, BLOSUM62) %>%
  drop_na()

#Convert CLASS to factor
clinvar_sub$CLASS <- as.factor(clinvar_sub$CLASS)

#Train/test split
set.seed(123)
train_index <- createDataPartition(clinvar_sub$CLASS, p = 0.8, list = FALSE)
train_data <- clinvar_sub[train_index, ]
test_data  <- clinvar_sub[-train_index, ]

#Fix NAs
fix_na <- function(df) {
  df %>% mutate(across(
    -CLASS,
    ~ if (is.numeric(.)) {
      ifelse(is.na(.), median(., na.rm = TRUE), .)
    } else {
      ifelse(is.na(.), names(sort(table(.), decreasing=TRUE))[1], .)
    }
  ))
}

train_data <- fix_na(train_data)
test_data  <- fix_na(test_data)

#️One-hot encoding
factor_cols <- c("Consequence", "IMPACT", "SIFT", "PolyPhen")

all_data <- bind_rows(train_data, test_data)

all_data_enc <- fastDummies::dummy_cols(
  all_data,
  select_columns = factor_cols,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

train_data_enc <- all_data_enc[1:nrow(train_data), ]
test_data_enc  <- all_data_enc[(nrow(train_data)+1):nrow(all_data_enc), ]

#matrices for xgboost
x_train <- as.matrix(train_data_enc %>% select(-CLASS))
y_train <- as.numeric(train_data_enc$CLASS) - 1

x_test  <- as.matrix(test_data_enc %>% select(-CLASS))
y_test  <- as.numeric(test_data_enc$CLASS) - 1



#class weights
class_table <- table(y_train)
weight_0 <- as.numeric(class_table[2] / sum(class_table))
weight_1 <- as.numeric(class_table[1] / sum(class_table))

weights <- ifelse(y_train == 0, weight_0, weight_1)
dtrain <- xgb.DMatrix(data = x_train, label = y_train, weight = weights)

#Train XGBoost with class weights
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)

xgb_v2 <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 0
)

#Predictions
dtest <- xgb.DMatrix(data = x_test, label = y_test)
xgb_probs <- predict(xgb_v2, dtest)

#Threshold tuning (default 0.5)
threshold <- 0.5
xgb_pred_class <- ifelse(xgb_probs >= threshold, 1, 0) %>% as.factor()

#Confusion matrix
conf_mat <- confusionMatrix(xgb_pred_class, factor(y_test, levels=c(0,1)), positive="1")
print(conf_mat)

#ROC curve & AUC
roc_obj <- roc(y_test, xgb_probs)
plot(roc_obj, main="XGBoost v2 ROC Curve", col="blue")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")

