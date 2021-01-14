# Classification: Logistic Regression, Decision Trees, Support Vector Machines, Naive Bayes Classifier, Artificial Neural Networks, k-Nearest Neighbor &
# Random Forest, Discriminant Analysis

library(dplyr)

# data

data = Ecdat::BudgetFood
data = na.omit(data)
str(data)

# k-Nearest Neighbor

data1 = data[,-ncol(data)]
str(data1)

dataset = sample(seq_len(nrow(data1)), nrow(data1) * 0.60)
train1 = data1[-dataset,]
train1_out = data1[dataset,]
train2 = data[-dataset, "sex"]
train2_out = data[dataset, "sex"]

knn_pred = knn(train1, train1_out, train2, k = 3)
knn_acc = mean(train2_out == knn_pred)
knn_acc

# Logistic Regression

dataset = sample(c(1:nrow(data)), nrow(data) * 80) # dataset
train = data[dataset,] # entrenamiento
test  = data[-dataset,] # prueba

log_reg = glm(formula = sex ~ ., data = train, family = "binomial")
lr_pred1 = predict(log_reg, type = "response")
lr_pred2 = predict(log_reg, newdata = test, type = "response")
class1 = ifelse(lr_pred1 <= 0.5, "man", "woman")
class2 = ifelse(lr_pred2 <= 0.5, "man", "woman")

lg_acc1 = mean(class1 == train$sex)
lg_acc2 = mean(class2 == test$sex)
lg_acc1
lg_acc2

# Decision Tree

library(rpart)

dec_tree = rpart(formula = sex ~ ., data = train, method = "class")
dt_pred1 = predict(dec_tree, type = "class")
dt_pred2 = predict(dec_tree, newdata = test, type = "class")
table_mat1 = table(train$sex, dt_pred1)
table_mat2 = table(test$sex, dt_pred2)
dt_acc1 = sum(diag(table_mat1)) / sum(table_mat1)
dt_acc2 = sum(diag(table_mat2)) / sum(table_mat2)
dt_acc1
dt_acc2

rpart.plot(dec_tree)

# Support Vector Machine

library(e1071)

svm_model = svm(formula = sex ~ ., data = train, cost = 10, kernel = "linear", type = "C-classification")
svm_pred1 = predict(svm_model)
svm_pred2 = predict(svm_model, newdata = test)
table_mat1 = table(train$sex, svm_pred1)
table_mat2 = table(test$sex, svm_pred2)
svm_acc1 = sum(diag(table_mat1)) / sum(table_mat1)
svm_acc2 = sum(diag(table_mat2)) / sum(table_mat2)
svm_acc1
svm_acc2

# Naive Bayes Classifier

library(e1071)

nb_model = naiveBayes(formula = sex ~ ., data = train)
nb_pred = predict(nb_model, newdata = test)
table_mat = table(test$sex, nb_pred)
nb_acc = sum(diag(table_mat)) / sum(table_mat)
nb_acc

# Artificial Neural Networks

library(neuralnet)

set.seed(123)
nn_model = neuralnet(formula = sex ~ ., data = train)
nn_pred = compute(nn_model, test)
nn_prob = nn_pred$net.result
nn_pred2 = ifelse(nn_prob <= 0.5, "man", "woman")

a = length(nn_pred2)

nn_acc = mean(nn_pred2 == data[1:a,"sex"])
nn_acc

plot(nn_model)

# Random Forest

library(randomForest)

rf_model = randomForest(formula = sex ~ ., data = train, ntree = 10, mtry = 2, importance = TRUE)
rf_pred1 = predict(rf_model)
rf_pred2 = predict(rf_model, newdata = test)
table_mat1 = table(train$sex, rf_pred1)
table_mat2 = table(test$sex, rf_pred2)
rf_acc1 = sum(diag(table_mat1)) / sum(table_mat1)
rf_acc2 = sum(diag(table_mat2)) / sum(table_mat2)
rf_acc1
rf_acc2

plot(rf_model)

# Quadratic Discriminant Analysis

library(MASS)

qda_model = qda(formula = sex ~ ., data = train)
qda_pred1 = predict(qda_model)
qda_pred2 = predict(qda_model, newdata = test)
table_mat1 = table(train$sex, qda_pred1$class)
table_mat2 = table(test$sex, qda_pred2$class)
qda_acc1 = sum(diag(table_mat1)) / sum(table_mat1)
qda_acc2 = sum(diag(table_mat2)) / sum(table_mat2)
qda_acc1
qda_acc2

# Analysis

results = data.frame(knn_acc, lg_acc2, dt_acc2, svm_acc2, nb_acc, nn_acc, rf_acc2, qda_acc2)
colnames(results) = c("k-Nearest Neighbor", "Logistic Regression", "Decision Tree", "Support Vector Machine", "Naive Bayes Classifier",
                      "Artificial Neural Networks", "Random Forest", "Quadratic Discriminant Analysis")
results
results1 = sort(round(results,4)*100, decreasing = T)
results1