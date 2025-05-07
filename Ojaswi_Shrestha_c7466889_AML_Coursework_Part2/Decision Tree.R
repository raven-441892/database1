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

#Decision Tree
#Load caret package for data partitioning and modeling
library(caret)

#Split data into training (80%) and testing (20%) sets
trainIndex<-createDataPartition(clean_bank_data$y,p=0.8,list=FALSE)
train_data<-clean_bank_data[trainIndex,]
test_data<-clean_bank_data[-trainIndex,]

#Convert target variable to factor for classification
train_data$y<-as.factor(train_data$y)
test_data$y<-as.factor(test_data$y)

#Check class distribution in training and testing sets
prop.table(table(train_data$y)) * 100
prop.table(table(test_data$y)) * 100

#Load rpart package to build decision tree
library(rpart)

#Fit decision tree model using training data
cart_fit<-rpart(y ~ ., data=train_data, method="class")

#Load rpart.plot package for visualizing the decision tree
library(rpart.plot)

#Plot the trained decision tree
rpart.plot(cart_fit)

#Predict class labels on the test data using the decision tree
dt_pred<-predict(cart_fit,test_data,type="class")

#Generate confusion matrix to evaluate predictions
confusionMatrix(data=dt_pred, reference = test_data$y, positive = "yes")

#Predict class probabilities for ROC analysis
dt_prob<-predict(cart_fit,test_data,type="prob")[,"yes"]

#Load pROC package for ROC curve and AUC calculation
library(pROC)

#Create ROC curve object for decision tree model
roc_obj<-roc(test_data$y,dt_prob,levels=c("no","yes"))

#Plot ROC curve and display AUC
plot(roc_obj, col = "pink", print.auc = TRUE)

#Calculate and display AUC value
auc_val <- auc(roc_obj)
auc_val

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
  tuneLength = 10       #Number of tuning parameter combinations to try
)

#Plot the best tuned decision tree
rpart.plot(cart_model$finalModel)

#Predict class labels using the tuned decision tree
tune_dt_pred<-predict(cart_model,test_data)

#Predict class probabilities for ROC analysis
tune_dt_prob<-predict(cart_model,test_data,type="prob")[,"yes"]

#Generate confusion matrix for tuned model predictions
confusionMatrix(data=tune_dt_pred, reference = test_data$y, positive = "yes")

#Create and plot ROC curve for the tuned model
tune_roc<-roc(test_data$y,tune_dt_prob,levels=c("no","yes"))
plot(tune_roc, col = "purple", print.auc = TRUE)

#Calculate and display AUC for the tuned model
auc_val <- auc(tune_roc)
auc_val