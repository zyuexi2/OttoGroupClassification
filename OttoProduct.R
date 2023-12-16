library(tidyverse)
library(vroom)
library(tidymodels)
library(lightgbm)
library(methods)
library(dplyr)

# Load data

train <- read.csv('/Users/cicizeng/Desktop/STA348/OttoProduct/train.csv', header = TRUE, stringsAsFactors = FALSE)
test <- read.csv('/Users/cicizeng/Desktop/STA348/OttoProduct/test.csv',header = TRUE, stringsAsFactors = FALSE)
train <- train[, -1]
test <- test[, -1]

target_value <- train[, ncol(train)]
target_value <- gsub('Class_', '', target_value, fixed = TRUE)
target_value <- as.integer(target_value) - 1  

features <- rbind(train[, -ncol(train)], test)
features <- as.matrix(features)

# Split indices for training and testing
trind <- seq_len(nrow(train))
teind <- (nrow(train) + 1):nrow(features)

# Create LightGBM datasets
lgb_train <- lgb.Dataset(data = features[trind, ], label = target_value)

# Set necessary parameters for LightGBM
params <- list(
  objective = "multiclass",
  metric = "multi_logloss",
  num_class = 9,
  num_threads = 8,
  learning_rate = 0.3,
  max_depth = 6
)
folds <- 50
bst_model <- lgb.cv(
  params = params,
  data = lgb_train,
  nfold = 5,
  nrounds = folds,
  early_stopping_rounds = 10
)

optimal_nrounds <- bst_model$best_iter

# Train the final model
final_model <- lgb.train(
  params = params,
  data = lgb_train,
  nrounds = optimal_nrounds
)

# Make predictions
pred <- predict(final_model, features[teind, ])
pred <- matrix(pred, nrow = 9, ncol = length(pred) / 9)
pred <- t(pred)

pred <- format(pred, digits = 2, scientific = FALSE)  # shrink the size of submission
submission <- data.frame(id = seq_len(nrow(pred)), pred)
colnames(submission) <- c('id', paste0('Class_', 1:9))
write.csv(submission, file = 'submission.csv', quote = FALSE, row.names = FALSE)
