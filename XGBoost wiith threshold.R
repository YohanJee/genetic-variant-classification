library(tidyverse)
library(caret)
library(fastDummies)
library(xgboost)
library(pROC)

#Define factor columns
factor_cols <- c("Consequence", "IMPACT", "SIFT", "PolyPhen")

#One-hot encode train & test
train_data_enc <- fastDummies::dummy_cols(
  train_data_fixed,
  select_columns = factor_cols[factor_cols %in% names(train_data_fixed)],
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

test_data_enc <- fastDummies::dummy_cols(
  test_data_fixed,
  select_columns = factor_cols[factor_cols %in% names(test_data_fixed)],
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

#Align test columns to train columns
missing_cols <- setdiff(names(train_data_enc), names(test_data_enc))
for(col in missing_cols) test_data_enc[[col]] <- 0
test_data_enc <- test_data_enc[, names(train_data_enc)]

#Prepare numeric matrices
x_train <- train_data_enc %>% select(-CLASS) %>% mutate_all(as.numeric) %>% as.matrix()
x_test  <- test_data_enc  %>% select(-CLASS) %>% mutate_all(as.numeric) %>% as.matrix()

y_train <- ifelse(train_data_enc$CLASS == "1", 1, 0)
y_test  <- ifelse(test_data_enc$CLASS  == "1", 1, 0)

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

#Compute class weights
weight_0 <- sum(y_train == 1) / length(y_train)
weight_1 <- sum(y_train == 0) / length(y_train)
weights <- ifelse(y_train == 1, weight_1, weight_0)

#Train XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)

xgb_v3 <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 1,
  weight = weights
)

#Predict probabilities
xgb_probs <- predict(xgb_v3, dtest)

#ROC curve and optimal threshold
roc_obj <- roc(y_test, xgb_probs)
auc_value <- auc(roc_obj)
plot(roc_obj, main = "XGBoost v3 ROC Curve", col = "blue")

#Optimal threshold using Youden's J
roc_coords <- coords(roc_obj, x = "best", best.method = "youden")
threshold <- roc_coords[["threshold"]]

cat("AUC:", auc_value, "\n")
cat("Optimal Threshold:", threshold, "\n")

#Predictions using optimal threshold
xgb_pred_class <- ifelse(xgb_probs >= threshold, 1, 0)
stopifnot(length(xgb_pred_class) == length(y_test))
xgb_pred_class <- factor(xgb_pred_class, levels = c(0,1))
y_test_factor  <- factor(y_test, levels = c(0,1))

conf_mat <- confusionMatrix(xgb_pred_class, y_test_factor, positive = "1")
print(conf_mat)




