#---------------------------------------------------
#         Logistic Regression with Elastic Net            
#---------------------------------------------------

#' Function to cross validate for all hyperparameters in Elastic Net.
#' 
#' @param mXTrain Matrix, explain variables.
#' @param vYTrain Vector, dependent binary variable.
#' @param lLambda list, lambda hyper-parameters.
#' @param lAlpha list, alpha hyper-parameters.
#' @param nFolds integer, number of folds.
#' 
#' @return lErrorMin list, minimum error obtained with the optimal lambda,
#' given the alpha value. From this, the optimal alpha can be obtained.
CV_lambda_alpha = function(mXTrain, vYTrain, lLambda, lAlpha, nfolds){
  # Initialize
  lErrorMin = list()

  # Perform validation
  for (i in 1:length(lAlpha)){
    Cv_result = cv.glmnet(mXTrain, vYTrain, alpha = lAlpha[i], 
                          lambda = lLambda, nfolds = nfolds,family = "binomial",type.measure = "class")

    # Get minimum error over all values of lambda
    lError = Cv_result$cvm
    # Find error of the optimal lambda, given alpha value
    dMinError = min(lError)
    # Get a list of minimum errors associated with each alpha, after fixing the optimal lambda
    lErrorMin[i] = dMinError
  }
  return(lErrorMin)
}


#' Function to compare changes in error rates with most and least optimal values of alphas.
#' 
#' @param mXTrain matrix, training data of explain variables.
#' @param vYTrain vector, training data of dependent variables.
#' @param dAlphaMin double, most optimal alpha.
#' @param dAlphaMax double, least optimal alpha.
#' @param lLambda list, full list of lamda parameters.
#' 
#' @return Cv_min, Cv_max cross-validation results using the least and most optimal alphas.
compare_alphas_cv = function(mXTrain, vYTrain, dAlphaMin, dAlphaMax, lLambda){
  Cv_min <- cv.glmnet(mXTrain, vYTrain, alpha = dAlphaMin, 
                      lambda = lLambda, nfolds = 10, family = "binomial",type.measure = "class")  
  Cv_max <- cv.glmnet(mXTrain, vYTrain, alpha = dAlphaMax, 
                      lambda = lLambda, nfolds = 10, family = "binomial",type.measure = "class")  
  par(mfrow = c(1,2))
  plot(Cv_min); plot(Cv_max)
  return(list(Cv_min, Cv_max))
}

#---------------------------------------------------
#            Decision Trees with Pruning           
#---------------------------------------------------

#' Function to build, plot pruned tree, cross-validate for optimal number of splits.
#' 
#' @param df_train dataframe, traning data.
#' 
#' @return ptree optimal pruned tree.
pruned_tree = function(df_train){
  tree <- rpart(fraudulent ~ ., data = df_train,method = 'class')
  
  # Get optimal crossvalidated cp value to prune the tree
  cp_optimal = tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
  
  # Plot pruned tree
  ptree = rpart(fraudulent ~ ., data = df_train,cp= cp_optimal,method = 'class')
  par(mfrow=c(1,1))
  fancyRpartPlot(ptree, uniform=TRUE,
                 main="Pruned Classification Tree")
  return(list(ptree, cp_optimal))
}

#---------------------------------------------------
#                    Bagging Manual            
#---------------------------------------------------

#' Function to predict for a bagged sample and get out-of-bag error.
#' 
#' @param df_train dataframe, training data.
#' @param df_test dataframe, test data.
#' @param iNTree integer, number of bagged trees
#' @param cp_optimal double, optimal cp value for the pruned tree
#' 
#' @return list(test_predicts,OOB_error) list, each elements 
#' contains each bag's predictions and out-of-bag errors
bagged_tree_manual = function(df_train, df_test){
  n = nrow(df_train)
  idx = sample(1:n, size=n, replace=TRUE)
  in_bag_set = df_train[idx,]
  out_bag_set = df_train[-idx,]
  tree =  rpart(fraudulent ~ ., data = in_bag_set, method = 'class')

  #Get out-of-bag error
  bag_predicts = predict(tree, out_bag_set, type = 'class')
  OOB_error =1- mean(bag_predicts == out_bag_set$fraudulent)
  
  #Get predictions on test set
  test_predicts = predict(tree, df_test, type = 'class')
  return(list(test_predicts,OOB_error))
}

#' Function to extract out-of-bag errors.
#' 
#' @param result output from bagged_tree_manual function.
#' @return OOB_error list of out-of-bag Error.
extract_OOB_error = function(result){
  OOB_error = result[[2]]
  return(OOB_error)
}

#' Function to extract lists of predictions of all bags.
#' 
#' @param result output from bagged_tree_manual function.
#' @return OOB_error list of out-of-bag Error.
extract_bag_pred = function(result){
  pred = result[[1]]
  return(pred)
}

#' Function to get the majority vote's prediction.
#' 
#' @param lPredictions list of predictions, output from extract_bag_pred function.
#' @param df_test dataframe, test data.
#' @return votes final predictions on the test set.
manual_predictions_bagging = function(lPredictions, df_test){
  #Get matrix of predictions
  mPredictions <- sapply (lPredictions, function (x) {
    length (x) <- length(lPredictions[[1]]); return (x)}) %>% as.matrix()
  
  #Get majority votes
  votes = apply(mPredictions, 1, Compose(table,
                                         function(i) i==max(i),
                                         which,
                                         names,
                                         function(i) paste0(i))) %>% as.double()
  return(votes)
}

#---------------------------------------------------
#              Bagging using Package            
#---------------------------------------------------

#' Function to make predictions using the bagging package.
#' 
#' @param df_train dataframe, training data.
#' @param df_test dataframe, test data.
#' @param iNTree integer, number of bagged trees
#' @param cp_optimal double, optimal cp value for the pruned tree
#' 
#' @return bag_pred_package predictions using the best votes of all bags
bagged_package=function(df_train, df_test, iNTree, cp_optimal){
  
  #Fit bagging model with pruned trees
  bag <- bagging(
    formula = fraudulent ~ .,
    data = df_train,
    nbagg = iNTree,   
    coob = FALSE,
    control = rpart.control(cp = cp_optimal)
  )
  
  # Get predictions
  bag_proba <- predict(bag, df_test)
  bag_pred_package <- ifelse(bag_proba > 0.5, 1, 0)
  return(bag_pred_package)
}


#---------------------------------------------------
#                 Model evaluations            
#---------------------------------------------------

#' Evaluate model predictions.
#' 
#' @param predictions vector, predictions.
#' @param vYTest vector, dependent variable.
#' 
#' @return accuracy, precision, recall, f_measure.
model_evaluate = function(predictions, vYTest){
  # Model evaluations
  accuracy= mean(predictions == vYTest)
  sensitivity=sensitivity(table(vYTest, predictions))
  specificity=specificity(table(vYTest, predictions))
  # False Positive Rate
  FP <- 1-specificity
  # True positive Rate
  TP <- sensitivity
  # False negative
  FN = 1-TP
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f_measure = (2 * precision * recall) / (precision + recall)
  return(list(accuracy, precision, recall, f_measure))
}


