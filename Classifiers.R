library(caret)

#1.
filename <- "BlogFeedback\\wine1.csv"
dataset <- read.csv(filename, header=FALSE)
colnames(dataset) <- c(
  "ClassId",
  "Alcohol",
  "Malic acid",
  "Ash",
  "Alcalinity of ash",  
  "Magnesium",
  "Total phenols",
  "Flavanoids",
  "Nonflavanoid phenols",
  "Proanthocyanins",
  "Color intensity",
  "Hue",
  "OD280/OD315",
  "Proline"
)

#2.1
head(dataset,n=15)

#2.2
dim(dataset)
sapply(dataset, class)

#2.3
dataset$ClassId <- factor(dataset$ClassId)
table(dataset$ClassId)
percentage <- prop.table(table(dataset$ClassId)) * 100
cbind(freq=table(dataset$ClassId), percentage=percentage)

#2.4
summary(dataset)
sapply(dataset[2:14], sd)
library(moments)
skew <- apply(dataset[2:14], 2, skewness)
print(skew)
kurt <- apply(dataset[2:14], 2, kurtosis)
print(kurt)

#3
# split input and output
x <- dataset[,2:14]
y <- dataset[,1]

#4.1
#histogramy
par(mfrow=c(3,5))
for(i in 2:14) {
  
  hist(x[,i-1], main=names(wine)[i])
}
# barplot for class breakdown
plot(y)

#4.2
correlations <- cor(dataset[2:14])
print(correlations)
install.packages("corrplot")
library(corrplot)
par(mfrow=c(1,1))
corrplot(correlations, method="circle")

#4.3
# boxplot for each attribute on one image
par(mfrow=c(3,5))
for(i in 2:14) {
  boxplot(x[,i-1], main=names(wine)[i])
}


#4.4
# box and whisker plots for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="ellipse", scales=scales)

#4.5
# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#5
colnames(dataset) <- make.names(colnames(dataset))
validationIndex <- createDataPartition(dataset$ClassId, p=0.75, list=FALSE)
validation <- dataset[-validationIndex,]
dataset <- dataset[validationIndex,]

#6
# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#6.1
# Naive Bayes
set.seed(7)
fit.nb <- train(ClassId~., data=dataset, method="nb", metric=metric, trControl=trainControl)

#6.2
# KNN
set.seed(7)
fit.knn <- train(ClassId~., data=dataset, method="knn", metric=metric, trControl=trainControl)

#6.3
# SVM
set.seed(7)
fit.svm <- train(ClassId~., data=dataset, method="svmRadial", metric=metric, trControl=trainControl)

#6.4
# Decision Tree
set.seed(7)
fit.cart <- train(ClassId~., data=dataset, method="rpart", metric=metric,trControl=trainControl)

#6.5
#Random Forest
set.seed(7)
fit.rf <- train(ClassId~., data=dataset, method="rf", metric=metric, trControl=trainControl)

#6.6
# Neural Network
set.seed(7)
fit.nnet <- train(ClassId~., data=dataset, method="nnet", metric=metric, trControl=trainControl)


#7
# summarize accuracy of models
results <- resamples(list(nb=fit.nb, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, nnet=fit.nnet))
summary(results)
dotplot(results)

#8
print(fit.svm)
predictions <- predict(fit.svm, validation)
predictions <- factor(predictions)
validation$ClassId <- factor(validation$ClassId)

#9
cm <- confusionMatrix(predictions, as.factor(validation$ClassId))
cm
tab <- cm$table
TP <- sum(tab[1,1])
TP
FN <- sum(tab[2:3,1])
FN
FP <- sum(tab[1,2:3])
FP
TN <- sum(tab[2:3,2:3])
TN
accuracy <- (TP+TN) / (TP+FN+FP+TN)
accuracy
TPR <- TP / (TP+FN)
TPR
SPEC = TN / (TN+FP)
SPEC
FPR <- FP/ (FP+TN)
FPR
FDR <- FP / (FP+TP)
FDR
POS <- TP/(TP+FP)
POS
NEG <- TN/(TN+FN)
NEG
F1 <-TP / (TP+0.5*(FP+FN))
F1
FBETA<-(1+(0.5^2)) * ((TPR*POS)/(((0.5^2)*TPR)+POS))
FBETA
MCC <- (TP*TN - FP*FN) / (sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
MCC