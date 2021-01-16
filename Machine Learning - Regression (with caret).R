data = stevedata::mm_nhis
str(data)

apply(is.na(data),2,sum)

library(caret)

control = trainControl(method = "LGOCV", p = 0.8, number = 10, savePredictions = T)

mlr_model = train(form = inc~., data = data, method = "lm", preProcess = "scale", trControl = control) # Linear Regression
ppr_model = train(form = inc~., data = data, method = "ppr", preProcess = "scale", trControl = control) # Projection Pursuit Regression
rlm_model = train(form = inc~., data = data, method = "rlm", preProcess = "scale", trControl = control) # Robust Linear Model
ss_model = train(form = inc~., data = data, method = "spikeslab", preProcess = "scale", trControl = control) # Spike and Slab Regression
spca_model = train(form = inc~., data = data, method = "superpc", preProcess = "scale", trControl = control) # Supervised Principal Component Analysis
l_model = train(form = inc~., data = data, method = "lasso", preProcess = "scale", trControl = control) # The lasso
knn_model = train(form = inc~., data = data, method = "knn", preProcess = "scale", trControl = control) # k-Nearest Neighbors

results = matrix(c(mlr_model$results$RMSE, ppr_model$results$RMSE, rlm_model$results$RMSE, ss_model$results$RMSE, spca_model$results$RMSE,
                     l_model$results$RMSE, knn_model$results$RMSE), byrow = T, ncol = 1, nrow = 7)
rownames(results) = c("Linear Regression", "Projection Pursuit Regression", "Robust Linear Model", "Spike and Slab Regression", "Supervised Principal Component Analysis",
                      "The lasso", "k-Nearest Neighbors")
colnames(results) = "RMSE"
results = as.data.frame(results)
rs = results[order(-results$RMSE),]
rs1 = sort(rownames(results$RMSE), decreasing = T)


results[order(results$RMSE),]


results1 = sort(round(results$RMSE,4)*100, decreasing = F)
results1

results = data.frame(min(mlr_model$results$RMSE), 
                     min(ppr_model$results$RMSE), 
                     min(rlm_model$results$RMSE), 
                     min(ss_model$results$RMSE), 
                     min(spca_model$results$RMSE),
                     min(l_model$results$RMSE), 
                     min(knn_model$results$RMSE))
colnames(results) = c("Linear Regression", "Projection Pursuit Regression", "Robust Linear Model", "Spike and Slab Regression", "Supervised Principal Component Analysis",
                      "The lasso", "k-Nearest Neighbors")
results1 = sort(round(results,0))
results1
mean(data$inc)