#------------------------------------
#          Set up           
#------------------------------------
# load packages
if(!require(pacman)){install.packages("pacman")}
p_load(tidyverse, simglm, rlist, latex2exp, 
       glmnet, knitr, formatR, devtools, glmnetUtils,rpart,
       caret,parallel,functional, ipred,rattle,rpart.plot)

# set seed
set.seed(1234)

# import manual functions
source("./dev/toolbox.r")

#------------------------------------
#          Data Preparation           
#------------------------------------
# Load data
cleaned_data <- read_csv("cleaned_data.csv")
df <- data.frame(cleaned_data) 
df[is.na(df)] <- 0

# Split train and test data
ind <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.7,0.3))
df_train  <- df[ind, ]
df_test   <- df[!ind, ]

mXTrain = subset(df_train, select=-c(fraudulent)) %>% as.matrix()
vYTrain = as.vector(df_train$fraudulent)    
mXTest = subset(df_test, select=-c(fraudulent)) %>% as.matrix()
vYTest = as.vector(df_test$fraudulent)    

#------------------------------------
#         Logistic Regression            
#------------------------------------
# Initialize
lLambda=10^seq(-10, 5, length.out = 50)
lAlpha = seq(0, 1, length.out = 50)
nfolds = 10

# Cross validate for optimal alpha and lambda
lResult= CV_lambda_alpha(mXTrain, vYTrain, lLambda, lAlpha, nfolds)

# Get most and least optimal alphas
dAlphaMin = lAlpha[which.min(lResult)] # Most optimal
dAlphaMax = lAlpha[which.max(lResult)] # Least optimal

#' Keeping fixed most and least optimal alphas, 
#' compare difference in cross-validated errors over lambdas.
Cv_res = compare_alphas_cv(mXTrain, vYTrain, dAlphaMin, dAlphaMax, lLambda)
Cv_min =Cv_res[[1]]
cat("best CV lambda", Cv_min$lambda.min)
cat("best CV lambda 1SE", Cv_min$lambda.1se)


# Prediction
vBeta=coef(Cv_min, s = "lambda.min")
probabilities <- Cv_min %>% predict(mXTest, type = "response")
log_predictions <- ifelse(probabilities > 0.5, 1, 0)

# Evaluation
log_evaluations = model_evaluate(log_predictions, vYTest)

#------------------------------------
#            Decision Tree            
#------------------------------------

# Get the cp value of optimal pruned tree using train data
fit = pruned_tree(df_train)
ptree = fit[[1]]
cp_optimal = fit[[2]]

# Predictions on test set
tree_predictions = predict(ptree, df_test, type = 'class')

# Evaluation
tree_evaluation = model_evaluate(tree_predictions, vYTest)

#------------------------------------
#      Bagging Manual Algorithm             
#------------------------------------
iNTree= 500

#Get predictions and our of bag errors for 500 bagged samples in parallel with 8 cores
lBagResult = mclapply(1: iNTree, function(i) {bagged_tree_manual(df_train, df_test)}, 
                      mc.cores = parallel::detectCores(), mc.set.seed = TRUE)

#extract list of OOB Error in parallel
lOOB_error = mclapply(1:length(lBagResult), function(i){extract_OOB_error(lBagResult[[i]])}, 
                      mc.cores = parallel::detectCores(), mc.set.seed = TRUE)

#extract list of predictions in parallel
lPredictions = mclapply(1:length(lBagResult), function(i){extract_bag_pred(lBagResult[[i]])}, 
                        mc.cores = parallel::detectCores(), mc.set.seed = TRUE)

# Get predictions
bagging_predictions_manual = manual_predictions_bagging(lPredictions, df_test)

# Evaluations
bag_evaluations_manual = model_evaluate(bagging_predictions_manual, vYTest)

#------------------------------------
#       Bagging Using Package             
#------------------------------------
# Predict on test set
bag_predictions = bagged_package(df_train, df_test, iNTree, cp_optimal)

# Evaluations
bag_evaluations_package = model_evaluate(bag_predictions, vYTest)







