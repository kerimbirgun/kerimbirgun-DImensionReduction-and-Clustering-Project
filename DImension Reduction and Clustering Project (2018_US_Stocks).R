library(dplyr);library(tidyverse)

d18 = read.csv("2018_Financial_Data.csv")
d18$year = 2018
names(d18)[224] = "Next_Year_Price_Var"

data = d18
data$year = as.factor(data$year)

# Assuming no financial measure could exactly equal to 0
# Converting all 0 cells to NA  (except variable "class")
# Class variable is the end result of growth 1 is positive 0 is negative

data1 = data %>%  select(-c("Class"))
data1[data1 == 0] <- NA
data = cbind(data1,data$Class)
data$`data$Class` = as.factor(data$`data$Class`)
names(data)[names(data) == "data$Class"] <- "Class"
glimpse(data)
dim(data)


# DATA CLEANING

  #counting NA's
na_table = as.data.frame(colSums(is.na(data)))
ggplot(na_table,aes(x=colSums(is.na(data)))) + geom_density()
sum(na_table)

  #removing variableS that are missing >15% of total row (>658 rows)
data_gathered = gather(data,key = "col", value= "value",
    -c("Stock.Name","Sector","Class","year","Next_Year_Price_Var"))
glimpse(data_gathered)  

removals = as.data.frame(data_gathered %>%  select(col,value) 
  %>% group_by(col) %>% summarise(totalna = sum(is.na(value))) %>%
   filter(totalna > 658))

removals = as.list(removals$col)
data2 = data[,-which(names(data) %in% removals)]

na_table = as.data.frame(colSums(is.na(data2)))
ggplot(na_table,aes(x=colSums(is.na(data2)))) + geom_density()

  #imputation of the remaining dataset
library(naniar)
gg_miss_upset(data2, 
              nsets = 10,
              nintersects = 10)

library(missRanger)
data3 = missRanger(data2, formula = . ~ ., pmm.k = 0L, maxiter = 10L,
     seed = NULL, verbose = 0, returnOOB = FALSE, case.weights = NULL,
     num.trees = 100)

class(data3);glimpse(data3);dim(data3)
write.csv(data3,"cleaned_data.csv")
  
  ###Prediction results without employing "PCA & CLUSTERING" ###

data3 = read.csv("cleaned_data.csv")
glimpse(data3);dim(data3)

library(ggplot2)
ggplot(data3, aes(x=Next_Year_Price_Var,y=Next_Year_Price_Var)) + geom_boxplot()
data3 %>% select(Next_Year_Price_Var) %>%  filter(Next_Year_Price_Var > 500)
# 8 Major Outliers requires to be eliminated 
# (also try to run models without eliminating outliers and you will notice the train-test rmse gap)
data3 = data3 %>%  filter(Next_Year_Price_Var < 500)
ggplot(data3, aes(x=Next_Year_Price_Var,y=Next_Year_Price_Var)) + geom_boxplot()
# after observing the boxplot revised it to <350
data3 = data3 %>%  filter(Next_Year_Price_Var < 350)
# in total I eliminated 15 rows (out of 4392) to minimize variance


  # splitting data 
data3 = data3 %>% select(-c("year","Stock.Name",'Class','operatingProfitMargin'))  

library(caret);library(lattice)
set.seed(150)
split <- createDataPartition(data3$Next_Year_Price_Var, 
                             p = 0.7,list = F)
data_train <- data3[split,]
data_test  <- data3[-split,]

dim(data_train);dim(data_test)

  # linear regression   (RMSE Train:46.8, Test:53.6)

model_lm = lm(Next_Year_Price_Var~.,data=data_train)
summary(model_lm);vif(model_lm)

pred_lm_tr = predict(model_lm)
rmse_lm_tr = sqrt(mean((pred_lm_tr-data_train$Next_Year_Price_Var)^2)); rmse_lm_tr

pred_lm_ts = predict(model_lm,newdata=data_test)
rmse_lm_ts = sqrt(mean((pred_lm_ts-data_test$Next_Year_Price_Var)^2)); rmse_lm_ts


  # random forest (RMSE Train:46.4, Test:46.9)

library(randomForest)
set.seed(100)
model_rf = randomForest(Next_Year_Price_Var~.,data=data_train,ntree=1000)

pred_rf_tr = predict(model_rf)
rmse_rf_tr = sqrt(mean((pred_rf_tr-data_train$Next_Year_Price_Var)^2)); rmse_rf_tr

pred_rf_ts = predict(model_rf,newdata=data_test)
rmse_rf_ts = sqrt(mean((pred_rf_ts-data_test$Next_Year_Price_Var)^2)); rmse_rf_ts


  # gradient boosting  (RMSE Train:36.9, Test:47.2)

library(gbm)
set.seed(617)
boosted_model = gbm(Next_Year_Price_Var~.,data=data_train,verbose = TRUE,shrinkage = 0.01,  
                    interaction.depth = 6,n.minobsinnode = 5, n.trees = 1500,cv.folds = 10)

pred_gb_tr = predict(boosted_model)
rmse_gb_tr = sqrt(mean((pred_gb_tr-data_train$Next_Year_Price_Var)^2)); rmse_gb_tr

pred_gb_ts = predict(boosted_model,newdata=data_test)
rmse_gb_ts = sqrt(mean((pred_gb_ts-data_test$Next_Year_Price_Var)^2)); rmse_gb_ts


  # XGBoost  (RMSE Train:44.8, Test:47.3)

require(xgboost)

# one hot encoding for "sector" categorical variable

#for train data
sector <- model.matrix(~Sector-1,data_train)
train_numeric = data_train %>% select(-c('Sector'))  
train_numeric = cbind(train_numeric,sector)

X_train = xgb.DMatrix(as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))))
y_train = train_numeric$Next_Year_Price_Var

#for test data
sector <- model.matrix(~Sector-1,data_test)
test_numeric = data_test %>% select(-c('Sector')) 
test_numeric = cbind(test_numeric,sector)

X_test = xgb.DMatrix(as.matrix(test_numeric %>% select(-c('Next_Year_Price_Var'))))
y_test = test_numeric$Next_Year_Price_Var


set.seed(100)
XGB1 <- xgboost(data = as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))), 
                    label = train_numeric$Next_Year_Price_Var,eta=0.01,nrounds = 750,
                    max_depth=2,booster = "gbtree",learning_rate = 0.005,early_stopping_rounds = 10,
                    objective = "reg:linear")   


predict_train = predict(XGB1, X_train)
residuals = y_train - predict_train
rmse_xgb_tr = sqrt(mean(residuals^2));rmse_xgb_tr


predicted = predict(XGB1, X_test)
residuals = y_test - predicted
rmse_xgb_ts = sqrt(mean(residuals^2));rmse_xgb_ts

glimpse(test_numeric)
glimpse(train_numeric)
glimpse(data_train)



################################################################################



  ## New Predictive Models with "PCA" ##


data3 = read.csv("cleaned_data.csv")
glimpse(data3);dim(data3)

# again eliminate outliers
data3 = data3 %>%  filter(Next_Year_Price_Var < 350)

# splitting data 

data3 = data3 %>% select(-c("year","Stock.Name",'Class'))  

library(caret);library(lattice)
set.seed(150)
split <- createDataPartition(data3$Next_Year_Price_Var, 
                             p = 0.7,list = F)
data_train <- data3[split,]
data_test  <- data3[-split,]


# drop outcome variable + 'Sector' variable since it is categorical
train = data_train %>% select(-c('Next_Year_Price_Var','Sector'))  
test = data_test %>% select(-c('Next_Year_Price_Var','Sector')) 

# Correlation Matrix
cor_matrix = data.frame(cor(train[,unlist(lapply(train, is.numeric))]))
write.csv(cor_matrix, 'cor_matrix.csv',row.names = F)

## warning ## 'operatingProfitMargin' variable yields NA 
## and therefore I can't measure Measuring Sampling Adequacy (MSA)
train = train %>% select(-c('operatingProfitMargin'))  
test = test %>% select(-c('operatingProfitMargin')) 


# Data Correlation Visualization  
      ## Can anyone find a solution for the scaling?
corr <- round(cor(train), 2)
col <- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = corr, col = col, symm = TRUE,margins = c(0,0) )


# Measuring Sampling Adequacy (MSA)  - we epxect this number to be at least 0.5
library(psych)
KMO(cor(train)) # overall MSA is 0.81 - quite fine!

# Barlett's Test of Sphericty
cortest.bartlett(cor(train),n = nrow(train))

# Scree Plot
library(FactoMineR)
pca_facto = PCA(train,graph = F)
library(factoextra)
fviz_eig(pca_facto,ncp=50,addlabels = T)


library(factoextra);library(gridExtra)
charts = lapply(1:6,FUN = function(x) fviz_contrib(pca_facto,choice = 'var',axes = x,title=paste('Dim',x)))
grid.arrange(grobs = charts,ncol=3,nrow=2)

# eigen values
pca_facto$eig  # goes above 70% variance after the 15th component
# To ensure that the factors represents the original variables sufficiently well,
# the total variance explained by factors should be greater than 70%.

              # based on scree plot 17 component seems a good option to move forward

# Finalizing Components
train_pca = prcomp(train,scale. = T)
train_pca = data.frame(train_pca$x[,1:17])
dim(train_pca)

test_pca = prcomp(test,scale. = T)
test_pca = data.frame(test_pca$x[,1:17])
dim(test_pca)


#### Prediction results with "PCA" ####

 # Adding our dependent variable and categorical variable "sector" back 
train_add = data_train %>% select(c('Next_Year_Price_Var','Sector'))  
test_add = data_test %>% select(c('Next_Year_Price_Var','Sector')) 

train_pca = cbind(train_pca,train_add)
test_pca = cbind(test_pca,test_add)
dim(train_pca);dim(test_pca)



# linear regression with PCA  (RMSE Train:47.8, Test:49.6) 

model_lm_pca = lm(Next_Year_Price_Var~.,data=train_pca)

pred_lm_tr_pca = predict(model_lm_pca)
rmse_lm_tr_pca = sqrt(mean((pred_lm_tr_pca-train_pca$Next_Year_Price_Var)^2)); rmse_lm_tr_pca

pred_lm_ts_pca = predict(model_lm_pca,newdata=test_pca)
rmse_lm_ts_pca = sqrt(mean((pred_lm_ts_pca-test_pca$Next_Year_Price_Var)^2)); rmse_lm_ts_pca


# random forest with PCA   (RMSE Train:47.8, Test:49.6)

library(randomForest)
set.seed(100)
model_rf_pca = randomForest(Next_Year_Price_Var~.,data=train_pca,ntree=1000)

pred_rf_tr_pca = predict(model_rf_pca)
rmse_rf_tr_pca = sqrt(mean((pred_rf_tr_pca-train_pca$Next_Year_Price_Var)^2)); rmse_rf_tr_pca

pred_rf_ts_pca = predict(model_rf_pca,newdata=test_pca)
rmse_rf_ts_pca = sqrt(mean((pred_rf_ts_pca-test_pca$Next_Year_Price_Var)^2)); rmse_rf_ts_pca


# gradient boosting with PCA  (RMSE Train:45.1, Test:49.6)

library(gbm)
set.seed(617)
boosted_model_pca = gbm(Next_Year_Price_Var~.,data=train_pca,verbose = TRUE,shrinkage = 0.01,  
                    interaction.depth = 3,n.minobsinnode = 5, n.trees = 1000,cv.folds = 10)

pred_gb_tr_pca = predict(boosted_model_pca)
rmse_gb_tr_pca = sqrt(mean((pred_gb_tr_pca-train_pca$Next_Year_Price_Var)^2)); rmse_gb_tr_pca

pred_gb_ts_pca = predict(boosted_model_pca,newdata=test_pca)
rmse_gb_ts_pca = sqrt(mean((pred_gb_ts_pca-test_pca$Next_Year_Price_Var)^2)); rmse_gb_ts_pca



# XGBoost with PCA   (RMSE Train:46.8, Test:49.2)
require(xgboost)


# one hot encoding for "sector" categorical variable
# add our one-hot encoded variable and convert the dataframe into a matrix

#for train data
sector <- model.matrix(~Sector-1,train_pca)
train_numeric = train_pca %>% select(-c('Sector'))  
train_numeric = cbind(train_numeric,sector)

X_train = xgb.DMatrix(as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))))
y_train = train_numeric$Next_Year_Price_Var

#for test data
sector <- model.matrix(~Sector-1,test_pca)
test_numeric = test_pca %>% select(-c('Sector')) 
test_numeric = cbind(test_numeric,sector)

X_test = xgb.DMatrix(as.matrix(test_numeric %>% select(-c('Next_Year_Price_Var'))))
y_test = test_numeric$Next_Year_Price_Var


set.seed(100)
XGB2 <- xgboost(data = as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))), 
                label = train_numeric$Next_Year_Price_Var,eta=0.01,nrounds = 750,
                max_depth=2,booster = "gbtree",learning_rate = 0.005,early_stopping_rounds = 10,
                objective = "reg:linear")   

predict_train = predict(XGB2, X_train)
residuals = y_train - predict_train
rmse_xgb_tr_pca = sqrt(mean(residuals^2));rmse_xgb_tr_pca


predicted = predict(XGB2, X_test)
residuals = y_test - predicted
rmse_xgb_ts_pca = sqrt(mean(residuals^2));rmse_xgb_ts_pca






################################################################################



#### 3rd Prediction with PCA + K-Means Clustering ####

#To cluster the dataset, we have to remove the outcome and factor variable

train_pca2 = train_pca
test_pca2 = test_pca


trainMinusDV = train_pca2 %>% select(-c('Next_Year_Price_Var','Sector'))  
testMinusDV = test_pca2 %>% select(-c('Next_Year_Price_Var','Sector')) 

# data normalizing

preproc = preProcess(trainMinusDV)
trainNorm = predict(preproc,trainMinusDV)
testNorm = predict(preproc,testMinusDV)

set.seed(1706)
km = kmeans(x = trainNorm,centers = 2,iter.max=10000,nstart=100)


# 2 Cluster is the optimal based on the silhoutte plot

library(cluster)
silhoette_width = sapply(2:10,FUN = function(x) pam(x = trainNorm,k = x)$silinfo$avg.width)
ggplot(data=data.frame(cluster = 2:10,silhoette_width),aes(x=cluster,y=silhoette_width))+
  geom_line(col='steelblue',size=1.2)+
  geom_point()+
  scale_x_continuous(breaks=seq(2,10,1))

# Two Clusters
set.seed(1706)
km = kmeans(x = trainNorm,centers = 2,iter.max=10000,nstart=100)


#Visualizing Clusters
library(psych)
temp = data.frame(cluster = factor(k_segments),
          factor1 = fa(trainNorm,nfactors = 2,rotate = 'varimax')$scores[,1],
          factor2 = fa(trainNorm,nfactors = 2,rotate = 'varimax')$scores[,2])
ggplot(temp,aes(x=factor1,y=factor2,col=cluster))+
  geom_point()



#### Prediction results with "PCA" with "Clustering" ####
k_segments = km$cluster
train_pca2 = cbind(train_pca2, k_segments)

set.seed(1706)
km2 = kmeans(x = testNorm,centers = 2,iter.max=10000,nstart=100)
k_segments = km2$cluster
test_pca2 = cbind(test_pca2, k_segments)  
glimpse(test_pca2)

# linear regression with PCA  (RMSE Train:47.8, Test:49.6) 

model_lm_pca2 = lm(Next_Year_Price_Var~.,data=train_pca2)

pred_lm_tr_pca2 = predict(model_lm_pca2)
rmse_lm_tr_pca2 = sqrt(mean((pred_lm_tr_pca2-train_pca2$Next_Year_Price_Var)^2)); rmse_lm_tr_pca2

pred_lm_ts_pca2 = predict(model_lm_pca2,newdata=test_pca2)
rmse_lm_ts_pca2 = sqrt(mean((pred_lm_ts_pca2-test_pca2$Next_Year_Price_Var)^2)); rmse_lm_ts_pca2


# random forest with PCA   (RMSE Train:47.8, Test:49.5)

library(randomForest)
set.seed(100)
model_rf_pca2 = randomForest(Next_Year_Price_Var~.,data=train_pca2,ntree=1000)

pred_rf_tr_pca2 = predict(model_rf_pca2)
rmse_rf_tr_pca2 = sqrt(mean((pred_rf_tr_pca2-train_pca2$Next_Year_Price_Var)^2)); rmse_rf_tr_pca2

pred_rf_ts_pca2 = predict(model_rf_pca2,newdata=test_pca2)
rmse_rf_ts_pca2 = sqrt(mean((pred_rf_ts_pca2-test_pca$Next_Year_Price_Var)^2)); rmse_rf_ts_pca2


# gradient boosting with PCA  (RMSE Train:45.1, Test:49.6)

library(gbm)
set.seed(617)
boosted_model_pca2 = gbm(Next_Year_Price_Var~.,data=train_pca2,verbose = TRUE,shrinkage = 0.01,  
                        interaction.depth = 3,n.minobsinnode = 5, n.trees = 1000,cv.folds = 10)

pred_gb_tr_pca2 = predict(boosted_model_pca2)
rmse_gb_tr_pca2 = sqrt(mean((pred_gb_tr_pca2-train_pca2$Next_Year_Price_Var)^2)); rmse_gb_tr_pca2

pred_gb_ts_pca2 = predict(boosted_model_pca2,newdata=test_pca2)
rmse_gb_ts_pca2 = sqrt(mean((pred_gb_ts_pca2-test_pca2$Next_Year_Price_Var)^2)); rmse_gb_ts_pca2


### Predict with XGBOOST ###  
require(xgboost)

# one hot encoding for "sector" categorical variable
# add our one-hot encoded variable and convert the dataframe into a matrix

#for train data
sector <- model.matrix(~Sector-1,train_pca2)
train_numeric = train_pca2 %>% select(-c('Sector'))  
train_numeric = cbind(train_numeric,sector)

X_train = xgb.DMatrix(as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))))
y_train = train_numeric$Next_Year_Price_Var

#for test data
sector <- model.matrix(~Sector-1,test_pca2)
test_numeric = test_pca2 %>% select(-c('Sector')) 
test_numeric = cbind(test_numeric,sector)

X_test = xgb.DMatrix(as.matrix(test_numeric %>% select(-c('Next_Year_Price_Var'))))
y_test = test_numeric$Next_Year_Price_Var


set.seed(100)
XGB3 <- xgboost(data = as.matrix(train_numeric %>% select(-c('Next_Year_Price_Var'))), 
         label = train_numeric$Next_Year_Price_Var,eta=0.01,nrounds = 750,
         max_depth=2,booster = "gbtree",learning_rate = 0.005,early_stopping_rounds = 10,
                    objective = "reg:linear")   

predict_train = predict(XGB3, X_train)
residuals = y_train - predict_train
rmse_xgb_tr_pca2 = sqrt(mean(residuals^2));rmse_xgb_tr_pca2


predicted = predict(bstDense, X_test)
residuals = y_test - predicted
rmse_xgb_ts_pca2 = sqrt(mean(residuals^2));rmse_xgb_ts_pca2


####################################################################################


# Model Performances

Without_PCA_Clustering = round(c(rmse_lm_ts,rmse_rf_ts,rmse_gb_ts,rmse_xgb_ts),1)
With_PCA = round(c(rmse_lm_ts_pca,rmse_rf_ts_pca,rmse_gb_ts_pca,rmse_xgb_ts_pca),1)
With_PCA_Clustering = round(c(rmse_lm_ts_pca2,rmse_rf_ts_pca2,rmse_gb_ts_pca2,rmse_xgb_ts_pca2),1)
model = c('Linear Regression','Random Forest','Gradient Boosting','XGBoosting')

Graph = as.data.frame(cbind(Without_PCA_Clustering,With_PCA,With_PCA_Clustering,model),stringsAsFactors = F)
Graph[,1:3] = as.numeric(unlist(Graph[,1:3]))
Graph[,4] = as.factor(Graph[,4])


Graph2 = gather(Graph,key = "col", value= "value",-c("model"))
Graph2;dim(Graph2)


ggplot(Graph2,aes(x=col,y=value,col=model))+
  geom_jitter(size=3)



