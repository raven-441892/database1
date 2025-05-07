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

#Check the proportion of each class in the training and test datasets
prop.table(table(train_data$y)) * 100
prop.table(table(test_data$y)) * 100

#Train a logistic regression model using 5-fold cross-validation
caret_glm_mod=train(
  form=y ~ .,             #Use all predictors to predict 'y'
  data=train_data,
  trControl=trainControl(method="cv",number=5),     #5-fold CV
  method="glm",     #Generalized Linear Model
  family="binomial"
)

#Display summary of the trained logistic regression model
summary(caret_glm_mod)

#Predict classes on the test data
predicted_test<-predict(caret_glm_mod, newdata = test_data)
predicted_test

#Generate confusion matrix to evaluate model performance
confusionMatrix(as.factor(predicted_test),test_data$y,positive = "yes")

#Threshold tuning: get predicted probabilities for class "yes"
predicted_probs <- predict(caret_glm_mod, newdata = test_data, type = "prob")[, "yes"]
predicted_probs

#Apply custom threshold of 0.3 to classify predictions
predicted_classes <- ifelse(predicted_probs > 0.3, "yes", "no")
predicted_classes

#Evaluate the model using the new threshold
confusionMatrix(as.factor(predicted_classes), test_data$y, positive = "yes")

#Load pROC for ROC curve analysis
library(pROC)

#Generate ROC curve based on predicted probabilities
roc_curve <- roc(test_data$y, predicted_probs)
roc_curve

#Calculate AUC (Area Under the Curve)
auc_value <- auc(roc_curve)
auc_value

#Plot ROC curve with AUC value in the title
plot(roc_curve, col = "green", main = paste("AUC =", round(auc_value, 3)))

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

#Evaluate the tuned model using a confusion matrix
confusionMatrix(as.factor(predicted_classes), test_data$y, positive = "yes")

#Generate and plot ROC curve for the tuned model
roc_curve <- roc(test_data$y, predicted_probs)
roc_curve
auc_value <- auc(roc_curve)
auc_value
plot(roc_curve, col = "blue", main = paste("AUC =", round(auc_value, 3)))




