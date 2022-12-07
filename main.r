# Setup -------------------------------------------------------------------
# load packages
if(!require(pacman)){install.packages("pacman")}
p_load(tidyverse, simglm, rlist, latex2exp, glmnet, knitr, formatR, devtools, glmnetUtils)

# set seed
set.seed(1234)

# import custom functions
source("./dev/toolbox.r")

# writing format objects
mytheme <- theme_bw() + theme(legend.position = "bottom")

# Import dataset
cleaned_data <- read_csv("cleaned_data.csv")
df <- data.frame(cleaned_data)
vFraudulent <- as.vector(df$fraudulent)      # y variable
mExplainVars <- as.matrix(subset(df, select=-c(fraudulent,1)))

#Elastic net
Cv_Enet <- cv.glmnet(mExplainVars, vFraudulent, alpha = 0.5, 
                       lambda = 10^seq(-10, 5, length.out = 50), nfolds = 10, standardize = TRUE)  
print(Cv_Enet$lambda.min)    
print(Cv_Enet$lambda.1se)   

# 
Cv_Enet$cvm  <- Cv_Enet$cvm^0.5
Cv_Enet$cvup <- Cv_Enet$cvup^0.5
Cv_Enet$cvlo <- Cv_Enet$cvlo^0.5
plot(Cv_Enet, ylab = "Root Mean-Squared Error") 
print(Cv_Enet$lambda.min)     
# Final run with best cross validated lambda
result <- glmnet(X, y, alpha = 0.5, lambda = Cv_Enet$lambda.min,
                 intercept = FALSE)  
result$beta


# Cross validate for optimal alpha
lAlpha = seq(0, 1, length.out = 50)
lErrorMin = list()
lLambda=10^seq(-10, 5, length.out = 50)
lLambdaMin = list()
for (i in 1:length(lAlpha)){
   Cv_result = cv.glmnet(mExplainVars, vFraudulent, alpha = lAlpha[i], 
                         lambda = lLambda, nfolds = 10, standardize = TRUE)
   # Get minimum error
   dMinError = min(Cv_result$cvm)
   lErrorMin[i] = dMinError
   
   # Get minimum lambda
   iIndex = which.min(Cv_result$cvm)
   lLambdaMin[i] = lLambda[iIndex]
   
   #matrix of errors over lamda and alpha
}
# Get minimum alpha

# Heatmap of RMSE over lambda and alpha
# Create and cross validate tree https://www.edureka.co/blog/implementation-of-decision-tree/ 


