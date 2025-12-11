library(randomForest)
library(pROC)
library(caret)
library(dplyr)

#Function to replace NA in numeric/factor columns
fix_na <- function(df) {
  df %>% mutate(across(
    !CLASS,
    ~ if (is.numeric(.)) {
      ifelse(is.na(.), median(., na.rm = TRUE), .)
    } else {
      ifelse(is.na(.), names(sort(table(.), decreasing = TRUE))[1], .)
    }
  ))
}

train_data_fixed <- fix_na(train_data)
test_data_fixed  <- fix_na(test_data)

train_data_fixed$CLASS <- as.factor(train_data_fixed$CLASS)
test_data_fixed$CLASS  <- as.factor(test_data_fixed$CLASS)

x_train <- train_data_fixed %>% select(-CLASS)
y_train <- train_data_fixed$CLASS

x_test  <- test_data_fixed %>% select(-CLASS)
y_test  <- test_data_fixed$CLASS

set.seed(123)

rf_v1 <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(x_train))), 
  importance = TRUE
)

rf_v1_probs <- predict(rf_v1, x_test, type = "prob")[,2]
rf_v1_pred  <- ifelse(rf_v1_probs > 0.5, "1", "0") %>% as.factor()

cat("=== RF V1: Basic Model ===\n")
confusionMatrix(rf_v1_pred, y_test, positive = "1")

#ROC object
rf_v1_roc <- roc(response = y_test, predictor = rf_v1_probs)

#Plot ROC curve
plot(
  rf_v1_roc,
  main = "ROC Curve - Random Forest v1 (Basic Model)",
  col = "blue",
  lwd = 3
)

#AUC value
rf_v1_auc <- auc(rf_v1_roc)
cat("AUC (RF v1):", rf_v1_auc, "\n")
