library(caret)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(ggplot2)
library(reshape2)

# Load the dataset
data <- read.csv(file.choose())
data$Rain <- as.factor(data$Rain)

# Split data into training and testing sets (70% train, 30% test)
set.seed(123)
trainIndex <- createDataPartition(data$Rain, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Define function to calculate performance metrics
calculate_metrics <- function(true, pred) {
  cm <- confusionMatrix(pred, true)
  accuracy <- cm$overall['Accuracy']
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  f1 <- 2 * (precision * recall) / (precision + recall)
  error_rate <- 1 - accuracy
  return(c(accuracy, precision, recall, f1, error_rate))
}

# Initialize results data frame
results <- data.frame(
  Model = c("KNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"),
  Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, ErrorRate = NA
)

# k-Nearest Neighbors
knn_pred <- knn(train = trainData[, -ncol(trainData)], test = testData[, -ncol(testData)], cl = trainData$Rain, k = 3)
knn_metrics <- calculate_metrics(testData$Rain, knn_pred)
results[1, 2:6] <- knn_metrics

# Support Vector Machine
svm_model <- svm(Rain ~ ., data = trainData, kernel = "linear", cost = 1)
svm_pred <- predict(svm_model, testData)
svm_metrics <- calculate_metrics(testData$Rain, svm_pred)
results[2, 2:6] <- svm_metrics

# Decision Tree
dt_model <- rpart(Rain ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
dt_metrics <- calculate_metrics(testData$Rain, dt_pred)
results[3, 2:6] <- dt_metrics

# Random Forest
rf_model <- randomForest(Rain ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, testData)
rf_metrics <- calculate_metrics(testData$Rain, rf_pred)
results[4, 2:6] <- rf_metrics

# Naive Bayes
nb_model <- naiveBayes(Rain ~ ., data = trainData)
nb_pred <- predict(nb_model, testData)
nb_metrics <- calculate_metrics(testData$Rain, nb_pred)
results[5, 2:6] <- nb_metrics

# Display the results
print(results)

# Visualization
results_long <- reshape2::melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Value")
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal()

