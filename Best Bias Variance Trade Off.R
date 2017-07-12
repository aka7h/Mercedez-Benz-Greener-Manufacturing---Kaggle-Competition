library(caret)
library(glmnet)

train <- read.csv('DMB/train.csv',header=T)
test <- read.csv('DMB/test.csv',header=T)

summary(train[,1:11])

col_names <- colnames(train)

#function to collect features with unique values equal to 1
drop_features <- function(d){
  cc <- colnames(d)
  c <- vector()
  for(i in cc){
    if(length(unique(d[,i])) == 1){
      c <- c(c,i)
    }    
  }
  return(c)
}

train_drop <- drop_features(train)
test_drop <- drop_features(test[,-2])
final_drop <- c(train_drop, test_drop)


train[,final_drop] <- NULL
test[,final_drop] <- NULL

#removing multicolinearity
require(corrplot)
t <- cor(train[,-c(1,3:10)])
highlyCorrelation <- findCorrelation(t,cutoff = 0.95, names = T)

train[,highlyCorrelation] <- NULL
test[,highlyCorrelation] <- NULL


image(as.matrix(train[,-c(1:10)]), main="Heatmap of binary data")

#We can see that there are few white lines. It will be best for the model if we just remove it.
#this we will take a look later.

test$y <- NA
train_test <- rbind.data.frame(train,test)
test$y <- NULL

#converting the character to interger
LETTERS702 <- tolower(c(LETTERS,sapply(LETTERS,function(x) paste0(x,LETTERS))))
cat_col <- c()
for(i in col_names){
  if(class(train[,i])=='factor'){
    cat_col <- append(cat_col,i,after = length(cat_col))
  }  
}

for(i in cat_col){
  u <- unique(train_test[,i])
  levels <- intersect(LETTERS702,u)
  label <- match(levels,LETTERS702)
  train_test[,i] <- as.character(factor(train_test[,i],levels = levels,labels = label))
  train_test[,i] <- as.integer(train_test[,i])
}

train <- train_test[1:4209,]
test <- train_test[4210:8418,]


#regression subset selection for Categorical values
library(leaps)

leaps <- regsubsets(y~X0+X1+X2+X3+X4+X5+X6+X8,data=train,method = c("exhaustive"), nbest=5)
plot(leaps, scale="adjr2", main = "Regression Subset Selection for Categorical values")
summary(leaps)

#the best model seems to be y~X0+X1+X2+X3

lm.model <- lm(y~X0+X1+X2+X3, data=train)
summary(lm.model)

train[,7:10] <- NULL
test[,7:10] <- NULL

ID <- test$ID
train$ID <- NULL
test$ID <- NULL
test$y <- NULL
test[] <- lapply(test,as.numeric)

#linear model
lm.model <- lm(y~.,train)
summary(lm.model) #0.5646


#Converting the intergers back to characters
test$y <- NA
train_test <- rbind.data.frame(train,test)
test$y <- NULL

for(i in 2:5){
  train_test[,i] <- as.factor(LETTERS702[train_test[,i]])
}

#label encoding 
dum <- dummyVars(~.,train_test[,2:5])
label.en <- data.frame(predict(dum,train_test[,2:5]))
tt <- data.frame('y'=train_test$y,label.en,train_test[,6:276])

ntrain <- tt[1:4209,]; ntest <- tt[4210:8418,]


#PCA

ptrain <- prcomp(ntrain[,-1], scale.= FALSE, center = FALSE)
ptest <- prcomp(ntrain[,-1], scale.= FALSE, center = FALSE)

stdev.train <- ptrain$sdev
variance.train <- stdev.train^2
propotion.variance.train <- variance.train/sum(variance.train)
plot(propotion.variance.train, type='b')
cumulative.train <- cumsum(propotion.variance.train)
plot(cumulative.train[1:50], type='b')

#from the summary i am taking the first 150 variables

tr <- data.frame('y'=ntrain$y,ptrain$x[,1:150])
te <- data.frame(ptest$x[,1:150])

#regularization

library(glmnet)
set.seed(1128)
x.tr <- model.matrix(y~.,tr)[,-1]; y.tr <- tr$y
x.te <- as.matrix(te)

#lasso regression
cv.glm <- cv.glmnet(x.tr,y.tr,nfolds = 10, type.measure = "mse", alpha=1)
plot(cv.glm)


#lambda values
cv.glm$lambda.min;cv.glm$lambda.1se

gnet.model <- glmnet(x.tr,y.tr, alpha = 0.5, lambda = cv.glm$lambda.min, standardize = TRUE, type.gaussian = "naive")
gnet.model
# plot(gnet.model, xvar = 'lambda',label = T)

predictt <- predict(gnet.model,x.tr, s=cv.glm$lambda.min)

#R2 function
r2_score <- function(a,p){
  R2 <- 1 - (sum((a-p)^2)/sum((a-mean(a))^2))
  return(R2)
}
errors <- function(a,p){
  return(sum(a-p))
}

print('r2 score')
r2_score(y.tr, predictt) #0.5775633
errors(y.tr,predictt)


test_prediction_lasso <- predict(gnet.model,x.te,s=cv.glm$lambda.min) #0.55183 in public leader board

# ##contineuing
# #Now Lets run XGBOOST
# library(xgboost)
# 
# xgtrain <- xgb.DMatrix(x.tr,label=y.tr)
# xgtest <- xgb.DMatrix(x.te)
# 
# 
# set.seed(1762)
# xgb_param <- list(colsample_bytree = 0.7, #how many variables to consider for each tree
#                   subsample = 0.7, #how much of the data to use for each tree
#                   booster = "gbtree",
#                   n_trees = 500,
#                   max_depth = 4, #how many levels in the tree
#                   eta = 0.03, #shrinkage rate to control overfitting through conservative approach
#                   eval_metric = "rmse",
#                   objective = "reg:linear",
#                   gamma = 0,
#                   base_score = mean(y.tr))
# 
# 
# xgbcv <- xgb.cv(params = xgb_param, nfold = 10,nrounds = 10000, print_every_n = 10,
#                 early_stopping_rounds = 20,data = xgtrain)
# xgbcv$params
# 
# xgb_model <- xgb.train(xgb_param,xgtrain,nrounds = xgbcv$best_iteration,print_every_n = 1)
# xgb_model
# xgb.plot.multi.trees(xgb_model)
# xgb.importance(colnames(xgtrain),model = xgb_model)[1:20]
# 
# xgb_prediction <- predict(xgb_model, xgtrain)
# 
# print('r2 score')
# r2_score(y.tr, xgb_prediction) #0.8034782
# errors(y.tr,xgb_prediction) #88.23136  #95.34962
# 
# xgb_pred_test <- predict(xgb_model,xgtest)# gave 0.55189


submission <- data.frame('ID'=ID,'y'=test_prediction_lasso)
colnames(submission) <- c('ID','y')
filename <- paste('ak_pca_glm_0.0498',format(Sys.time(),"%Y%m%d%H%M%s"),sep = '_')
# write.csv(submission,paste0(filename,'.csv',collapse = ''),row.names = FALSE)


#xgb v1 - gamma = 0.5, nrounds=60,train r2=0.6108834, test r2 = 0.55189
#xgb v2 - gamma = 0.5, nrounds=67, train r2 = 0.6161198, test r2 = 0.54987
#xgb v2 - gamma = 1, nrounds=2000, train r2 = 0.8034782, test r2 = 0.31148
#xgb v2 - gamma = 0, nrounds=2000, train r2 = 0.6142225, test r2 = 0.55183


#alpha=1 PCA, lambda.min, train r2 = 0.5775633, public r2 = 0.55183, private r2 = 0.54490
