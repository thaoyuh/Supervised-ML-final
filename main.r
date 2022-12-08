# Setup -------------------------------------------------------------------
# load packages
if(!require(pacman)){install.packages("pacman")}
p_load(tidyverse, simglm, rlist, latex2exp, glmnet, knitr, formatR, devtools, glmnetUtils,rpart)

# set seed
set.seed(1234)

# import custom functions
source("./dev/toolbox.r")

# writing format objects
mytheme <- theme_bw() + theme(legend.position = "bottom")

#----------------Preparing data
#MAKE FUNCTION
# Import dataset
cleaned_data <- read_csv("cleaned_data.csv")
df <- data.frame(cleaned_data) 
df[is.na(df)] <- 0
ind <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.7,0.3))
df_train  <- df[ind, ]
df_test   <- df[!ind, ]

mXTrain = subset(df_train, select=-c(fraudulent)) %>% as.matrix()
vYTrain = as.vector(df_train$fraudulent)      # y variable

mXTest = subset(df_test, select=-c(fraudulent)) %>% as.matrix()
vYTest = as.vector(df_test$fraudulent)      # y variable

#----------------------------------Logistic Regression
#--------------CV for optimal alpha and lamda
# MAKE FUNCTION
# Cross validate for optimal alpha


lLambda=10^seq(-10, 5, length.out = 50)
lAlpha = seq(0, 1, length.out = 50)
nfolds = 10

CV_lambda_alpha = function(mExplainVars, vFraudulent, lLambda, lAlpha, nfolds){
  # Initialize
  lErrorMin = list()

  # mError = matrix(NA, nrow=length(lLambda), ncol=length(lAlpha)) #matrix with nrows = number of lamda and ncols = number of alpha
  
  # Perform validation
  for (i in 1:length(lAlpha)){
    Cv_result = cv.glmnet(mExplainVars, vFraudulent, alpha = lAlpha[i], 
                          lambda = lLambda, nfolds = nfolds,family = "binomial",type.measure = "class")
    
    # Get minimum error
    lError = Cv_result$cvm
    #mError[,i] = lError
    dMinError = min(lError)
    lErrorMin[i] = dMinError
    
    # # Get minimum lambda
    # iIndex = which.min(Cv_result$cvm)

  }
  # List of return values
  
  return(lErrorMin)
}


lResult= CV_lambda_alpha(mXTrain, vYTrain, lLambda, lAlpha, nfolds)

lErrorMin=lResult[[1]]

# Get minimum and maximum alpha
# alpha=1 is lasso regression (default) and alpha=0 is ridge regression
iIndexMin = which.min(lErrorMin)
iIndexMax = which.max(lErrorMin)
dAlphaMin = lAlpha[iIndexMin] #1
dAlphaMax = lAlpha[iIndexMax] #0


# MAKE FUNCTION
#Elastic net using the optimal alpha value
Cv_min <- cv.glmnet(mXTrain, vYTrain, alpha = dAlphaMin, 
                     lambda = 10^seq(-10, 10, length.out = 50), nfolds = 10, family = "binomial",type.measure = "class")  
Cv_max <- cv.glmnet(mXTrain, vYTrain, alpha = dAlphaMax, 
                    lambda = 10^seq(-10, 10, length.out = 50), nfolds = 10, family = "binomial",type.measure = "class")  
print(Cv_min$lambda.min)    
print(Cv_min$lambda.1se)   

par(mfrow = c(1,2))
plot(Cv_min); plot(Cv_max)

# Final run with best cross validated lambda
coef(Cv_min, s = "lambda.min")

### MAKE FUNCTION
#Plot the minimum misclassification rate over Alphas
par(mfrow = c(1,1))
plot(x =lAlpha, y=lErrorMin,yaxt='n', xlab= "Alpha", ylab= "", col="red", pch=10)
ylabel <- seq(min(unlist(lErrorMin)), max(unlist(lErrorMin)), by = 0.0001) %>%round( digits = 4)
axis(2, at = ylabel,las=0)
title(ylab="Minimum Classification Error", line=2.7, cex.lab=1)

#-----------------Prediction and evaluations Logistic regression
# Summarize the model
summary(Cv_min)
# Make predictions
probabilities <- Cv_min %>% predict(mXTest, type = "response")
log_predictions <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
mean(log_predictions == vYTest)

# Confusion matrix
confusionMatrix(table(vYTest, log_predictions))

#calculate sensitivity
sensitivity=sensitivity(table(vYTest, log_predictions))

#calculate specificity
specificity=specificity(table(vYTest, log_predictions))

# False Positive Rate
FPR <- 1-specificity
# True positive Rate
TPR <- sensitivity


#----------------------DECISION TREE
# Create and cross validate tree https://www.edureka.co/blog/implementation-of-decision-tree/ 

library("rpart")
library("rattle")
library(rpart.plot)
library(RColorBrewer)
library(caret)

tree <- rpart(fraudulent ~ ., data = df_train,method = 'class')

rpart.plot(tree)

# 10-fold CV with printcp
printcp(tree)

plotcp(tree)

#Get pruned tree
ptree = rpart(fraudulent ~ ., data = df_train,cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"],method = 'class')
fancyRpartPlot(ptree, uniform=TRUE,
                 main="Pruned Classification Tree")

# Get classification error: Root node error x Rel error
classification_err= 0.4934*0.43324
classification_err

tree_predictions <- predict(ptree, df_test, type = 'class')

# Model accuracy
mean(tree_predictions == vYTest)

# Confusion matrix
confusionMatrix(table(vYTest, tree_predictions))

#calculate sensitivity
sensitivity=sensitivity(table(vYTest, log_predictions))

#calculate specificity
specificity=specificity(table(vYTest, log_predictions))

# False Positive Rate
FPR <- 1-specificity
# True positive Rate
TPR <- sensitivity



