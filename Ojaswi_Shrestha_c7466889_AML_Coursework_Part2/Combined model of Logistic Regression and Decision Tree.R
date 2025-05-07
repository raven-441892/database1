# Load the dataset
bank<-read.csv("C:\\Users\\NEW IT WIZARD\\OneDrive\\Desktop\\Applied Machine Learning\\Part1 project\\Bank Marketing Campaign.csv")
bank

# Display structure, dimensions, and summary of the dataset
str(bank)
dim(bank)
summary(bank)

# Load mice package for missing value analysis
library(mice)

# Check missing data pattern
md.pattern(bank)

# Calculate missing values with percent
missing_amount<-colSums(is.na(bank))
missing_percent<-(missing_amount/nrow(bank))*100
missing_percent

# Calculate total missing percentage
total_missing<-sum(missing_amount)
total_data<-ncol(bank)*nrow(bank)
total_missing_percent<-(total_missing/total_data)*100
total_missing_percent


# Remove rows with missing values
clean_bank_data<-na.omit(bank)

# Number of rows after removing missing values
nrow(clean_bank_data)

# Original number of rows
nrow(bank)

# Check missing data pattern again
md.pattern(clean_bank_data)

# Identify numeric columns
numeric_column <-sapply(clean_bank_data,is.numeric)
numeric_data<-clean_bank_data[,numeric_column]

# Function to detect outliers
is_outlier<-function(a){
  Q1<-quantile(a,.25)
  Q3<-quantile(a,.75)
  iqr<-Q3-Q1
  low<-Q1-1.5*iqr
  up<-Q3+1.5*iqr
  any(a<low | a>up)
}

# Find columns with outliers
outlier_columns <- names(numeric_data)[sapply(numeric_data, is_outlier)]
outlier_columns

# Function to replace outliers with median
median_imputation<-function(a){
  Q1<-quantile(a,.25)
  Q3<-quantile(a,.75)
  iqr<-Q3-Q1
  down<-Q1-1.5*iqr
  up<-Q3+1.5*iqr
  median_num<-median(a)
  a[a<down | a>up] <-median_num
  return (a)
}

# Function to cap outliers
cap <-function(a){
  Q1<-quantile(a,.25)
  Q3<-quantile(a,.75)
  iqr<-Q3-Q1
  down<-Q1-1.5*iqr
  up<-Q3+1.5*iqr
  
  a[a<down]<-down
  a[a>up]<-up
  return(a)
}

# Columns to apply outlier handling
cap_columns<-c("age","balance","duration")
med_columns<-c("campaign","pdays","previous")

# Apply median imputation for selected columns
for(col in med_columns){
  clean_bank_data[[col]]<-median_imputation(clean_bank_data[[col]])
}

# Apply capping for selected columns
for(col in cap_columns){
  clean_bank_data[[col]]<-cap(clean_bank_data[[col]])
}

#Logistic regression
#Load the caret package for model training and evaluation
library(caret)

#Split the data into 80% training and 20% testing
trainIndex<-createDataPartition(clean_bank_data$y,p=0.8,list=FALSE)
train_data<-clean_bank_data[trainIndex,]
test_data<-clean_bank_data[-trainIndex,]

#Convert the target variable to factor for classification
train_data$y<-as.factor(train_data$y)
test_data$y<-as.factor(test_data$y)

#Train a logistic regression model using 5-fold cross-validation
caret_glm_mod=train(
  form=y ~ .,             #Use all predictors to predict 'y'
  data=train_data,
  trControl=trainControl(method="cv",number=5),     #5-fold CV
  method="glm",     #Generalized Linear Model
  family="binomial"
)

#Predict classes on the test data
predicted_test<-predict(caret_glm_mod, newdata = test_data)
predicted_test

#Threshold tuning: get predicted probabilities for class "yes"
predicted_probs <- predict(caret_glm_mod, newdata = test_data, type = "prob")[, "yes"]

#Apply custom threshold of 0.3 to classify predictions
predicted_classes <- ifelse(predicted_probs > 0.3, "yes", "no")

#Model tuning
#Load glmnet package
library(glmnet)

#Prepare data for glmnet: convert to matrix form (excluding intercept column
x <- model.matrix(y ~ ., data = train_data)[,-1]
y <- train_data$y

#Train a penalized logistic regression model (Lasso/Ridge) with cross-validation
set.seed(123)   # For reproducibility
tuned_model <- train(
  x = x,
  y = y,
  method = "glmnet",       #Elastic net regularization
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC",      #Optimize based on AUC
  tuneLength = 10      #Number of tuning parameter combinations to try
)

#Prepare test data in matrix form for glmnet
x_test <- model.matrix(y ~ ., data = test_data)[,-1]

#Predict probabilities using the tuned glmnet model
predicted_probs <- predict(tuned_model, newdata = x_test, type = "prob")[, "yes"]

#Apply threshold of 0.3 to get predicted classes
predicted_classes <- ifelse(predicted_probs > 0.3, "yes", "no")


#Decision tree
#Load rpart package to build decision tree
library(rpart)

#Fit decision tree model using training data
cart_fit<-rpart(y ~ ., data=train_data, method="class")

#Predict class labels on the test data using the decision tree
dt_pred<-predict(cart_fit,test_data,type="class")

#Predict class probabilities for ROC analysis
dt_prob<-predict(cart_fit,test_data,type="prob")[,"yes"]

#Set up cross-validation and ROC-based evaluation for tuning
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary)

#Train decision tree model with tuning (searching over 10 combinations)
cart_model <- train(
  y ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  metric = "ROC",      
  tuneLength = 10      #Number of tuning parameter combinations to try
)

#Predict class labels using the tuned decision tree
tune_dt_pred<-predict(cart_model,test_data)

#Predict class probabilities for ROC analysis
tune_dt_prob<-predict(cart_model,test_data,type="prob")[,"yes"]

#Combined model of LR and DT
#Predict probabilities from decision tree model on train and test datasets
train_dt_prob<-predict(cart_model,train_data,type="prob")[,"yes"]
test_dt_prob<-predict(cart_model,test_data,type="prob")[,"yes"]

#Predict probabilities from tuned logistic regression model on train and test datasets
glm_train_prob<-predict(tuned_model, newdata = model.matrix(y ~ ., train_data)[,-1], type = "prob")[, "yes"]
glm_test_prob<-predict(tuned_model, newdata = model.matrix(y ~ ., test_data)[,-1], type = "prob")[, "yes"]

#Combine decision tree and logistic regression probabilities as features for training
combined_train<-data.frame(
  dt_probability=train_dt_prob,      # Decision tree probabilities
  glm_probability=glm_train_prob,    # Logistic regression probabilities
  y=train_data$y                     # Actual class labels
)

# Combine decision tree and logistic regression probabilities as features for testing
combined_test<-data.frame(
  dt_probability=test_dt_prob,
  glm_probability=glm_test_prob,
  y=test_data$y
)

#Train a logistic regression model on combined predictions (stacked model)
combined_model<-train(
  y ~ .,
  data=combined_train,
  method="glm",
  family="binomial",
  trControl=trainControl(method="cv", number = 5)
)

#Predict probabilities from the stacked model on the test data
combined_prob<-predict(combined_model, newdata = combined_test, type = "prob")[,"yes"]

#Apply threshold of 0.3 to classify probabilities as 'yes' or 'no'
combined_pred <- factor(ifelse(combined_prob > 0.3, "yes", "no"), levels = levels(combined_test$y))

#Evaluate the stacked model performance using confusion matrix
confusionMatrix(combined_pred,combined_test$y, positive="yes")

#Load pROC package for ROC curve and AUC calculation
library(pROC)

#Generate ROC curve for the stacked model
combined_roc <- roc(combined_test$y, combined_prob)

#Plot ROC curve and show AUC
plot(combined_roc, col = "", print.auc=TRUE)

#Calculate and return AUC value
auc(combined_roc)

#Load glmnet for regularized logistic regression
library(glmnet)

#Ensure target variable is factor
combined_train$y <- as.factor(combined_train$y)
combined_test$y <- as.factor(combined_test$y)

#Train a tuned logistic regression model using glmnet on the combined features
tuned_combined_model <- train(
  y ~ .,
  data = combined_train,
  method = "glmnet",
  trControl = trainControl(
    method = "cv",         #5-fold cross-validation
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary    #Use ROC metric
  ),
  metric = "ROC",      
  tuneLength = 10         #Try 10 combinations of hyperparameters
)

#Predict probabilities on test set using the tuned stacked model
tuned_combined_prob <- predict(tuned_combined_model, newdata = combined_test, type = "prob")[,"yes"]

#Classify predictions based on 0.3 threshold
tuned_combined_pred <- factor(ifelse(tuned_combined_prob > 0.3, "yes", "no"), levels = levels(combined_test$y))

#Evaluate the tuned stacked model using confusion matrix
confusionMatrix(tuned_combined_pred, combined_test$y, positive = "yes")

#Generate ROC curve for tuned stacked model
tuned_combined_roc <- roc(combined_test$y, tuned_combined_prob)

#Plot ROC curve and print AUC
plot(tuned_combined_roc, col = "purple", print.auc = TRUE)

#Calculate and display AUC value
auc(tuned_combined_roc)
