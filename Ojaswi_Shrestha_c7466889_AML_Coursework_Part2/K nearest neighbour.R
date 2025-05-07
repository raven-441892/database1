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

#Load caret library for data partitioning and model training
library(caret)

#Create training and testing partitions (80% training, 20% testing)
trainIndex<-createDataPartition(clean_bank_data$y,p=0.8,list=FALSE)
train_data<-clean_bank_data[trainIndex,]
test_data<-clean_bank_data[-trainIndex,]

#Extract true labels for test data
test_label<-as.factor(clean_bank_data$y[-trainIndex])

#Set up 5-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5)

#Train KNN model using cross-validation
knnFit <- train(y ~ ., data = train_data, method = "knn", trControl = ctrl)
knnFit

#Predict class labels on the test data
knnPredict <- predict(knnFit,newdata = test_data )
knnPredict

#Generate confusion matrix for KNN predictions
confusionMatrix(knnPredict,test_label, positive="yes")

#Predict class probabilities for ROC analysis
knnPredict <- predict(knnFit,newdata = test_data , type="prob")
knnPredict

#Load pROC package for ROC curve and AUC calculation
library(pROC)

#Generate ROC curve and calculate AUC
res.roc <- roc(test_label, knnPredict[,1])
auc <-res.roc$auc
auc

#Plot ROC curve
plot.roc(res.roc, print.auc = TRUE, col="red")

#Set up tuning grid for K (number of neighbors)
tuneGrid <- expand.grid(k = seq(10, 20, by = 2))

#Redefine control settings for tuning with ROC as performance metric
ctrl <- trainControl(
       method = "cv",       # Cross-validation
       number = 5,          # 5 folds
       classProbs = TRUE,    # Enable class probabilities
       summaryFunction = twoClassSummary    # Use AUC for evaluation
   )

#Train KNN model with tuning over specified k values
knnFit <- train(
       y ~ ., 
       data = train_data,     
       method = "knn",
       trControl = ctrl,
       tuneGrid = tuneGrid,
       metric = "ROC"       # Optimize based on ROC
   )

#Display best model
knnFit

#Predict class labels using the tuned KNN model
knnPredict <- predict(knnFit,newdata = test_data )
knnPredict

#Evaluate predictions with a confusion matrix
confusionMatrix(knnPredict,test_label)

#Predict class probabilities with tuned model
knnPredict <- predict(knnFit,newdata = test_data , type="prob")
knnPredict

#Generate ROC curve and calculate AUC for the tuned model
res.roc <- roc(test_label, knnPredict[,1])
plot.roc(res.roc, print.auc = TRUE, col="red")
auc <-res.roc$auc
auc