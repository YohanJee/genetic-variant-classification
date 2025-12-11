library(xgboost)
library(caret)
library(dplyr)
library(pROC)

#Replace NA values
fix_na <- function(df) {
  df %>% mutate(across(
    everything(),
    ~ if (is.numeric(.)) {
      ifelse(is.na(.), median(., na.rm=TRUE), .)
    } else {
      ifelse(is.na(.), names(sort(table(.), decreasing=TRUE))[1], .)
    }
  ))
}

train_data_fixed <- fix_na(train_data)
test_data_fixed  <- fix_na(test_data)

#Ensure CLASS is factor
train_data_fixed$CLASS <- as.factor(train_data_fixed$CLASS)
test_data_fixed$CLASS  <- as.factor(test_data_fixed$CLASS)

#Encode categorical features as numeric
cat_cols <- c("Consequence", "IMPACT", "SIFT", "PolyPhen")
for(col in cat_cols){
  levels_train <- levels(train_data_fixed[[col]])
  train_data_fixed[[col]] <- as.integer(factor(train_data_fixed[[col]], levels=levels_train))
  test_data_fixed[[col]]  <- as.integer(factor(test_data_fixed[[col]], levels=levels_train))
}

#Split features and labels
x_train <- as.matrix(train_data_fixed %>% select(-CLASS))
y_train <- as.numeric(train_data_fixed$CLASS) - 1

x_test  <- as.matrix(test_data_fixed %>% select(-CLASS))
y_test  <- as.numeric(test_data_fixed$CLASS) - 1

#Train XGBoost model
set.seed(123)
xgb_model <- xgboost(
  data = x_train,
  label = y_train,
  nrounds = 200,
  max_depth = 6,
  eta = 0.1,
  objective = "binary:logistic",
  eval_metric = "auc",
  verbose = 0
)

#Predict probabilities and classes
xgb_probs <- predict(xgb_model, x_test)
xgb_pred  <- ifelse(xgb_probs >= 0.5, 1, 0) %>% as.factor()

#Confusion matrix
conf_mat <- confusionMatrix(xgb_pred, factor(y_test, levels=c(0,1)), positive = "1")
print(conf_mat)

#ROC curve and AUC
roc_obj <- roc(y_test, xgb_probs)
plot(roc_obj, main="XGBoost ROC Curve", col="darkgreen")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")
