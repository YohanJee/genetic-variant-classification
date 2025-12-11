library(randomForest)
library(pROC)
library(caret)
library(dplyr)

#Fix NAs in numeric/factor columns
fix_na <- function(df) {
  df %>% mutate(across(
    -CLASS,   # exclude CLASS
    ~ if (is.numeric(.)) {
      ifelse(is.na(.), median(., na.rm = TRUE), .)
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

#Split into X and y
x_train <- train_data_fixed %>% select(-CLASS)
y_train <- train_data_fixed$CLASS

x_test  <- test_data_fixed %>% select(-CLASS)
y_test  <- test_data_fixed$CLASS

#Class-weighted Random Forest + Threshold Tuning
# Class weights for imbalance
class_weights <- c("0" = 1, "1" = 3)

set.seed(123)
rf_v3 <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(x_train))),
  importance = TRUE,
  classwt = class_weights
)

#Predict probabilities
rf_v3_probs <- predict(rf_v3, x_test, type = "prob")[,2]

#ROC curve and AUC
roc_obj <- roc(y_test, rf_v3_probs)
plot(roc_obj, main="RF v3 ROC Curve", col="blue")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")

#Optimal threshold (Youden's J statistic)
opt_coords <- coords(roc_obj, "best", best.method="youden", ret=c("threshold","sensitivity","specificity"))

#Extract numeric threshold
threshold <- as.numeric(opt_coords["threshold"])
cat("Optimal Threshold:", threshold, "\n")

#Predictions using optimal threshold
rf_v3_pred <- ifelse(rf_v3_probs >= threshold, "1", "0") %>% factor(levels=c("0","1"))

#Confusion matrix
confusionMatrix(rf_v3_pred, y_test, positive="1")

