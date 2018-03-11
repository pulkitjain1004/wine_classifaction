# Pulkit Jain, UIN 625005181
library(tree)
library(class)
library(boot)
library(e1071)
library(randomForest)
library(gbm)
library(MASS)

# III
# read and split data
data1 <- read.csv("C:/Users/Pulkit Jain/Desktop/ISEN__/613 Engg Data Analysis/Final/Dataset1.csv")
data2 <- read.csv("C:/Users/Pulkit Jain/Desktop/ISEN__/613 Engg Data Analysis/Final/Dataset2.csv")
data1$quality <- factor(data1$quality)
summary(data1$quality)

set.seed(1004)
train <- sort(sample(nrow(data1), 600, replace = F))

data1_train <- data1[train,]
data1_test <- data1[-train,]

# III.1 logistic regression

logistic.fit1 <- glm(quality~., data=data1_train, family=binomial)
summary(logistic.fit1)

# citric acid, alcohol and color don't appear to be affecting quality
# fixed acidity, residual sugar & total sulfur dioxide are mild influencers
# rest do appear to be influencing the quality

logistic.probs1 <- predict(logistic.fit1, data1_test, type = "response") 
logistic.pred1 <- rep(0, 257)
logistic.pred1[logistic.probs1>0.5] <- 1
table(Predicted = logistic.pred1, Actual = data1_test$quality)
mean(logistic.pred1 == data1_test$quality)

# III.2 tree

tree1 <- tree(quality~., data1_train)
plot(tree1)
text(tree1, pretty=0)


tree.pred1 <- predict(tree1, data1_test, type="class")  
table(Predicted = tree.pred1, Actual = data1_test$quality)
mean(tree.pred1 == data1_test$quality)

# tree pruning

set.seed(1004)
cv.tree1 <- cv.tree(tree1, FUN= prune.misclass)
plot(cv.tree1$size, cv.tree1$dev, type= "b")

cv.tree1$dev
cv.tree1$size
min(cv.tree1$dev)
which.min(cv.tree1$dev)
best_tree_size <- cv.tree1$size[which.min(cv.tree1$dev)]
best_tree_size

prune1 <- prune.misclass(tree1, best = best_tree_size)
plot(prune1)
text(prune1, pretty= 0)
# the first split is made on alcohol (shows its importance), 
# the other variables are volatile acidity, total sulfur dioxide, sulphates and chlorides


prune.pred1 <- predict(prune1, data1_test, type="class")
table(Predicted = prune.pred1, Actual = data1_test$quality)
mean(prune.pred1 == data1_test$quality)

# III.3



# KNN


# white is 0, red is 1

# train independent variables
train_var <- data1_train[,-13]
train_var$col <- as.character(train_var$col)
train_var$col[train_var$col == "white"] <- 0
train_var$col[train_var$col == "red"] <- 1

# test independent variables
test_res <- data1_test[,-13]
test_res$col <- as.character(test_res$col)
test_res$col[test_res$col == "white"] <- 0
test_res$col[test_res$col == "red"] <- 0

# train dependent variables
train_res <- data1_train$quality

err <- matrix(0,1,50)
for (i in 1:50){
set.seed(1004)
  results = knn.cv(train_var, k=i, cl = train_res )
  err[i] = 1-mean(results == train_res)
    }

plot(1:50, err, type = "b")
best_knn = which.min(err)
best_knn
# KNN works best when K( no. of nearest neighbours) is 11

knn.pred1 <- knn(train_var, test_res, train_res, k = best_knn)
table(Predicted = knn.pred1, Actual = data1_test$quality)
mean(knn.pred1 == data1_test$quality)

# SVM

aa = data.frame(train_var, train_res)
colnames(aa)[13] = "quality"

svmfit1 = svm(quality~., aa, kernel= "linear", cost=.1, scale = F)
summary(svmfit1)

set.seed(1004)
tune.out <- tune(svm, quality~., data = aa, kernel = "linear",
                  ranges = list(cost=c(.001, .01, .1, 1, 5, 10, 100) ))
summary(tune.out)

tune.out$best.parameters
bestmod = tune.out$best.model


bb = data1_test
bb$col <- as.character(bb$col)
bb$col[bb$col == "white"] <- 0
bb$col[bb$col == "red"] <- 1


ypred = predict(bestmod, bb)

table(Predicted = ypred, bb$quality)
mean(ypred == bb$quality)

# 
svmfit.radial <- svm(quality~., data=aa, kernal = "radial", 
                     gamma=1, cost = 10 ) 
summary(svmfit.radial)

set.seed(1004)
tune.out.radial <- tune(svm, quality~., data=aa, kernal = "radial", 
                        ranges = list(cost = 10^(seq(-1,3)), 
                                      gamma = 0.5*(seq(1,5)) ) )

summary(tune.out.radial)

tune.out.radial$best.parameters
tune.out.radial$best.performance

bestmod_rad = tune.out.radial$best.model

ypred_rad = predict(bestmod_rad, bb)

table(Predicted = ypred_rad,  Actual = bb$quality)
mean(ypred_rad == bb$quality)

# polynomial

svmfit.poly2 <- svm(quality~., data=aa, kernal = "polynomial", 
                    degree=2, cost = 10 ) 
summary(svmfit.poly2)

set.seed(1004)
tune.out.poly <- tune(svm, quality~., data=aa, kernal = "polynomial", 
                      ranges = list(cost = 10^(seq(-1,3)), 
                                    degree = c(2,3,4,5,10)) )
summary(tune.out.poly)

tune.out.poly$best.parameters
tune.out.poly$best.performance

bestmod_poly <- tune.out.poly$best.model

ypred_poly = predict(bestmod_poly, bb)

table(Predicted = ypred_poly,  Actual = bb$quality)
mean(ypred_rad == bb$quality)

# Bagging

set.seed(1004)
bag.wine <- randomForest(quality~., data=data1_train, mtry = 12, ntree =100,
                         importance=T)
bag.wine

yhat.bag <- predict(bag.wine, newdata = data1_test )
table(Predicted = yhat.bag,  Actual = data1_test$quality)
mean(yhat.bag == data1_test$quality)


err.bag = matrix(0,1,196)

for(i in 1:196){
  
  
  set.seed(1004)
  bag.wine <- randomForest(quality~., data=data1_train, mtry = 12, ntree =i+4,
                           importance=T)
  # bag.wine
  
  yhat.bag <- predict(bag.wine, newdata = data1_test )
  table(yhat.bag, data1_test$quality)
  err.bag[i]= 1- mean(yhat.bag == data1_test$quality)
}

plot(5:200, err.bag, type="l")
# ntree 100 is good enough


# Random Forest

err.rf = matrix(0,1,11)

for(i in 1:11){

set.seed(5)
rf.wine <- randomForest(quality~., data=data1_train, mtry = i, ntree =100,
                         importance=T)
yhat.rf <- predict(rf.wine, newdata = data1_test )
table(Predicted = yhat.rf, Actual =  data1_test$quality)
err.rf[i] = 1- mean(yhat.rf == data1_test$quality)
}
plot(1:11, err.rf, type="b")

which.min(err.rf)
no_pred = which.min(err.rf)


err.rf2 = matrix(0,1,196)

for(i in 1:196){
  
  
  set.seed(1004)
  rf.wine <- randomForest(quality~., data=data1_train, mtry = no_pred, ntree =i+4,
                           importance=T)
  # bag.wine
  
  yhat.rf <- predict(rf.wine, newdata = data1_test )
  
  err.rf2[i]= 1- mean(yhat.rf == data1_test$quality)
}

plot(5:200, err.rf2, type="l")
# ntree 100 is good enough


rf.wine <- randomForest(quality~., data=data1_train, mtry = no_pred, 
                        ntree =100,  importance=T)
importance(rf.wine)
varImpPlot(rf.wine)
graphics.off()
yhat.rf <- predict(rf.wine, newdata = data1_test )
table(Predicted = yhat.rf,  Actual = data1_test$quality)
mean(yhat.rf == data1_test$quality)

# Boosting

set.seed(1004)
boost.wine <- gbm(quality~., data = data1_train, distribution = "multinomial"
                   ,n.trees = 5000, interaction.depth = 4)
graphics.off()
# multinomial distribution has been used, which is generalized form of binomial
# due to some error in binomial syntax

summary(boost.wine)

yhat.boost <- predict(boost.wine, newdata = data1_test, n.trees = 5000)

yyhat.boost <- apply(yhat.boost, 1, which.max)

yyhat.boost[yyhat.boost==1] = 0
yyhat.boost[yyhat.boost==2] = 1
table(Predicted = yyhat.boost, Actual =  data1_test$quality)
mean(yyhat.boost == data1_test$quality)

# 

# lda
lda.fit <- lda(quality~., data=data1_train)
lda.pred <- predict(lda.fit, data1_test)
names(lda.pred)

table(Predicted = lda.pred$class,  Actual = data1_test$quality)
mean(lda.pred$class == data1_test$quality)

# qda

qda.fit <- qda(quality~., data=data1_train)
qda.pred <- predict(qda.fit, data1_test)
names(qda.pred)

table(Predicted = qda.pred$class,  Actual = data1_test$quality)
mean(qda.pred$class == data1_test$quality)


# check variance of independent variables

var(data1)
apply(data1[,-12], 2, var)
# variance are not comparable
# shall not use LDA

# check normality of data
graphics.off()
par(mfrow = c(2,3) )
qqnorm(data1$fix[data1$quality==0])
qqline(data1$fix[data1$quality==0])

qqnorm(data1$vol[data1$quality==0])
qqline(data1$vol[data1$quality==0])

qqnorm(data1$cit[data1$quality==0])
qqline(data1$cit[data1$quality==0])

qqnorm(data1$res[data1$quality==0])
qqline(data1$res[data1$quality==0])

qqnorm(data1$chl[data1$quality==0])
qqline(data1$chl[data1$quality==0])

qqnorm(data1$fre[data1$quality==0])
qqline(data1$fre[data1$quality==0])

graphics.off()

par(mfrow = c(2,3) )
qqnorm(data1$fix[data1$quality==1])
qqline(data1$fix[data1$quality==1])

qqnorm(data1$vol[data1$quality==1])
qqline(data1$vol[data1$quality==1])

qqnorm(data1$cit[data1$quality==1])
qqline(data1$cit[data1$quality==1])

qqnorm(data1$res[data1$quality==1])
qqline(data1$res[data1$quality==1])

qqnorm(data1$chl[data1$quality==1])
qqline(data1$chl[data1$quality==1])

qqnorm(data1$fre[data1$quality==1])
qqline(data1$fre[data1$quality==1])

# III.4
# 
# 

# III.5 train model on whole data

rf.wine.final <- randomForest(quality~., data=data1, mtry = no_pred
                              , ntree =100, importance=T)
importance(rf.wine.final)
varImpPlot(rf.wine.final)

# IV Predict on Dataset2
yhat.rf.final <- predict(rf.wine.final, newdata = data2)
yhat.rf.final

# 