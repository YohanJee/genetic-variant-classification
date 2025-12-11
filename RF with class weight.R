fix_na <- function(df) {
  df %>% mutate(across(
    -CLASS,   # exclude CLASS from cleaning
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

class_counts <- table(y_train)
min_class <- min(class_counts)

rf_v2 <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(x_train))),
  importance = TRUE,
  sampsize = rep(min_class, 2)
)

rf_v2_probs <- predict(rf_v2, x_test, type = "prob")[,2]
rf_v2_pred  <- ifelse(rf_v2_probs > 0.5, "1", "0") %>% as.factor()

cat("=== RF V2: Class-weighted ===\n")
conf_mat <- confusionMatrix(rf_v2_pred, y_test, positive = "1")
print(conf_mat)

roc_obj <- roc(y_test, rf_v2_probs)
plot(roc_obj, main="RF V2: ROC Curve", col="blue")
auc_value <- auc(roc_obj)
cat("AUC:", auc_value, "\n")
