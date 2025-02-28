library(tidyverse)
library(ggplot2)
library(skimr)
library(dplyr)
library(GGally)
library(caret)

#install.packages("caret") #this may take a while


# load hotel data
hotel_data<- read.csv("D://Direct Downloads//PredictiveAnalytics//hotel_data.csv", header= T)
hotel_data$X <- NULL
hotel_data$country <- NULL
hotel_data$cases_info <- NULL
hotel_data$assigned_room_type<- NULL

summaryStats <- skim(hotel_data)
summaryStats

hotel_data <- hotel_data %>% mutate_at(c("is_canceled","is_repeated_guest","hotel","customer_type","deposit_type","reserved_room_type","meal","market_segment","distribution_channel"), as.factor) 

#2. rename resonse 
hotel_data$is_canceled<-fct_recode(hotel_data$is_canceled, cancel = "1",notcancel = "0")

#3. relevel response
hotel_data$is_canceled<- relevel(hotel_data$is_canceled, ref = "cancel")

#make sure levels are correct
levels(hotel_data$is_canceled)

dummies_model<-dummyVars(is_canceled~.,data=hotel_data)
predictors_dummy<- data.frame(predict(dummies_model, newdata = hotel_data))

#recombine predictors including dummy variables with response
hotel_data <- cbind(is_canceled=hotel_data$is_canceled, predictors_dummy) 

#create training and testing
set.seed(99)
index <- createDataPartition(hotel_data$is_canceled, p = .8,list = FALSE)
hotel_data_train <-hotel_data[index,]
hotel_data_test <- hotel_data[-index,]

# Train model with preprocessing & cv
#install.packages("randomForest")
library(randomForest)  
library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

set.seed(8)
model_rf <- train(is_canceled~.,
                  data = hotel_data_train,
                  method = "rf",
                  tuneGrid= expand.grid(mtry = c(1, 3,6,9)),
                  trControl=trainControl(method = 'cv',number = 5,
                                         ## Estimate class probabilities
                                         classProbs = TRUE,
                                         #needed to get ROC
                                         summaryFunction = twoClassSummary
                  ),
                  metric="ROC")

#stop parallel processing
stopCluster(cl)
registerDoSEQ()

model_rf
plot(model_rf)
model_rf$bestTune
plot(varImp(model_rf))
plot(varImp(model_rf), top=20)

#First, get the predicted probabilities of the test data.
predprob_rf<-predict(model_rf , hotel_data_test, type="prob")

library(ROCR)
pred_rf<- prediction(predprob_rf$cancel, hotel_data_test$is_canceled,label.ordering =c("notcancel","cancel"))
perf_rf <- performance(pred_rf, "tpr", "fpr")
plot(perf_rf, colorize=TRUE)


#Get the AUC
auc_rf<-unlist(slot(performance(pred_rf, "auc"), "y.values"))
auc_rf
