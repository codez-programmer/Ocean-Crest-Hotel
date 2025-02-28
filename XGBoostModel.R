library(caret)
library(tidyverse)
library(skimr)

# load hotel data
hotel_data_boost<- read.csv("D://Direct Downloads//PredictiveAnalytics//hotel_data.csv", header= T)
hotel_data_boost$X <- NULL
hotel_data_boost$country <- NULL
hotel_data_boost$cases_info <- NULL
hotel_data_boost$assigned_room_type<- NULL

summaryStats <- skim(hotel_data_boost)
summaryStats

hotel_data_boost <- hotel_data_boost %>% mutate_at(c("is_canceled","is_repeated_guest","hotel","customer_type","deposit_type","reserved_room_type","meal","market_segment","distribution_channel"), as.factor) 

case_data_predictors<-select(hotel_data_boost,-is_canceled)
dummies_model<-dummyVars(~ ., data = case_data_predictors)
#provide only predictors that are now converted to dummy variables
predictors_dummy<- data.frame(predict(dummies_model, newdata = case_data_predictors)) 

#recombine predictors including dummy variables with response
hotel_data_boost <- cbind(is_canceled=hotel_data_boost$is_canceled, predictors_dummy) 

#2. rename resonse 
hotel_data_boost$is_canceled<-fct_recode(hotel_data_boost$is_canceled, cancel = "1",notcancel = "0")

#3. relevel response
hotel_data_boost$is_canceled<- relevel(hotel_data_boost$is_canceled, ref = "cancel")

#make sure levels are correct
levels(hotel_data_boost$is_canceled)


set.seed(99)
index <- createDataPartition(hotel_data_boost$is_canceled, p = .8,list = FALSE)
hotel_data_boost_train <-hotel_data_boost[index,]
hotel_data_boost_test <- hotel_data_boost[-index,]

#install.packages("xgboost")
library(xgboost)

#install.packages("doParallel")
library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

set.seed(8)
model_gbm <- train(is_canceled~.,
                   data = hotel_data_boost_train,
                   method = "xgbTree",
                   # provide a grid of parameters
                   tuneGrid = expand.grid(
                     nrounds = c(50,150),
                     eta = c(0.025, 0.09),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   trControl= trainControl(method = "cv",
                                           number = 5,
                                           classProbs = TRUE,
                                           summaryFunction = twoClassSummary),
                   metric = "ROC"
)
#stop parallel processing
stopCluster(cl)
registerDoSEQ()


pre_hotel_data<- read.csv("D://Direct Downloads//PredictiveAnalytics//prehotel_data.csv", header= T)
pre_hotel_data$X <- NULL
pre_hotel_data <- pre_hotel_data %>% mutate_at(c("is_repeated_guest","hotel","customer_type","deposit_type","reserved_room_type","meal","market_segment","distribution_channel"), as.factor) 

#Convert to dummy variables using the same dummies model of the training data 
pre_hotel_data<- data.frame(predict(dummies_model, newdata = pre_hotel_data)) 

case_holdoutprob<- predict(model_gbm, pre_hotel_data, type="prob")

case_holdout_scored<- cbind(pre_hotel_data, case_holdoutprob$cancel)
case_holdout_scored[1:3,]
write.csv(case_holdout_scored, file = "C:\\Users\\Kwaku\\Downloads\\case_holdout_scored.csv", row.names = FALSE)


plot(model_gbm)
model_gbm$bestTune
#only print top 10 important variables
plot(varImp(model_gbm), top=15)

#First, get the predicted probabilities of the test data.
predprob_xg<-predict(model_gbm , hotel_data_boost_test, type="prob")

library(ROCR)
pred_xg<- prediction(predprob_xg$cancel, hotel_data_boost_test$is_canceled,label.ordering =c("notcancel","cancel"))
perf_xg <- performance(pred_xg, "tpr", "fpr")
plot(perf_xg, colorize=TRUE)


#Get the AUC
auc_xg<-unlist(slot(performance(pred_xg, "auc"), "y.values"))
auc_xg





#install.packages("SHAPforxgboost")
library(SHAPforxgboost)

Xdata<-as.matrix(select(hotel_data_boost_train,-is_canceled)) # change data to matrix for plots

# Calculate SHAP values
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

# SHAP importance plot for top 15 variables
shap.plot.summary.wrap1(model_gbm$finalModel, X = Xdata, top_n = 10)

#example partial dependence plot
p <- shap.plot.dependence(
  shap, 
  x = "TRAN_AMT", 
  color_feature = "CUST_AGE", 
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) +
  ggtitle("TRAN_AMT")
print(p)
