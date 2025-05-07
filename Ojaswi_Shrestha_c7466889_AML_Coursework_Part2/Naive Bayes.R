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

#Naive Bayes
#Load caret package for data partitioning and model training
library(caret)

#Create training and testing datasets (80% training, 20% testing)
trainIndex<-createDataPartition(clean_bank_data$y,p=0.8,list=FALSE)
train_data<-clean_bank_data[trainIndex,]
test_data<-clean_bank_data[-trainIndex,]

#Convert the target variable to factor for classification
train_data$y<-as.factor(train_data$y)
test_data$y<-as.factor(test_data$y)

# Set up training control with 5-fold cross-validation, upsampling, and AUC as metric
control <- trainControl(
  method = "cv", 
  number = 5, 
  sampling = "up",       # Upsample the minority class to balance the data
  classProbs = TRUE, 
  summaryFunction = twoClassSummary)

#Train the Naive Bayes model using the training data
nb_model <- train(
  y ~ ., 
  data = train_data,
  method = "naive_bayes",     #Specify Naive Bayes classifier
  trControl = control,
  metric = "ROC"             #Optimize the model based on ROC
)

#Predict class labels on the test data
nb_predictions <- predict(nb_model, test_data)

#Predict class probabilities on the test data
nb_probabilities <- predict(nb_model, test_data, type = "prob")

#Evaluate model performance using confusion matrix
confusionMatrix(nb_predictions, test_data$y, positive = "yes")

#Load pROC library to plot ROC curve and compute AUC
library(pROC)

#Generate ROC object for the predicted probabilities
roc_obj <- roc(test_data$y, nb_probabilities$yes, levels = c("no", "yes"))

#Plot ROC curve and print AUC
plot(roc_obj,print.auc = TRUE, col = "cyan")

#Calculate and display AUC value
auc_val <- auc(roc_obj)
auc_val


#Retrain Naive Bayes model with tuning (5 combinations of hyperparameters)
nb_model <- train(
  y ~ ., 
  data = train_data,
  method = "naive_bayes",
  trControl = control,
  metric = "ROC",
  tuneLength=5        #Try 5 different tuning parameter settings
)

#Predict class labels using the tuned model
nb_predictions <- predict(nb_model, test_data)

#Predict class probabilities using the tuned model
nb_probabilities <- predict(nb_model, test_data, type = "prob")

#Evaluate tuned model performance using confusion matrix
confusionMatrix(nb_predictions, test_data$y, positive = "yes")

#Generate and plot ROC curve for the tuned model
roc_obj <- roc(test_data$y, nb_probabilities$yes, levels = c("no", "yes"))
plot(roc_obj,print.auc = TRUE, col = "darkgreen")

#Calculate and display AUC value for the tuned model
auc_val <- auc(roc_obj)
auc_val

