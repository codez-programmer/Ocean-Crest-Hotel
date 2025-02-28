library(caret)
library(tidyverse)

ht_data<- read.csv("D://Direct Downloads//PredictiveAnalytics//hotel_data.csv", header= T)
ht_data$X <- NULL
ht_data$country <- NULL
ht_data$cases_info <- NULL
ht_data$assigned_room_type<- NULL

library(skimr)
summaryStats <- skim(ht_data)
summaryStats

ht_data <- ht_data %>% mutate_at(c("is_canceled","is_repeated_guest","hotel","deposit_type","customer_type","reserved_room_type","meal","market_segment","distribution_channel"), as.factor) 

#2. rename resonse 
ht_data$is_canceled<-fct_recode(ht_data$is_canceled, cancel = "1",notcancel = "0")

#3. relevel response
ht_data$is_canceled<- relevel(ht_data$is_canceled, ref = "cancel")

#make sure levels are correct
levels(ht_data$is_canceled)

dummies_model<-dummyVars(is_canceled~.,data=ht_data)
predictors_dummy<- data.frame(predict(dummies_model, newdata = ht_data))

#recombine predictors including dummy variables with response
ht_data <- cbind(is_canceled=ht_data$is_canceled, predictors_dummy)

set.seed(99) #set random seed
index <- createDataPartition(ht_data$is_canceled, p = .8,list = FALSE)
ht_data_train <-ht_data[index,]
ht_data_test <- ht_data[-index,]


#install.packages("rpart")
library(rpart)
library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

set.seed(12)
ht_data_model <- train(is_canceled~.,
                     data =ht_data_train,
                     method = "rpart",
                     tuneGrid = expand.grid(cp=seq(0.01,0.2,length=5)),
                     trControl=trainControl(method = 'cv',number = 5,
                                            ## Estimate class probabilities
                                            classProbs = TRUE,
                                            #needed to get ROC
                                            summaryFunction = twoClassSummary),
                     metric="ROC") 
#stop parallel processing
stopCluster(cl)
registerDoSEQ()

ht_data_model
plot(ht_data_model)

#install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(ht_data_model$finalModel,type=5)

#Step 3 get predictons
predictions<- predict(ht_data_model,ht_data_test,type= "prob")

#step 4 get AUC
library(ROCR)
ht_data_test_prob<- prediction(predictions$cancel, ht_data_test$is_canceled, label.ordering = c("notcancel","cancel"))
perf <- performance(ht_data_test_prob, "tpr", "fpr")
plot(perf, colorize=TRUE)
slot(performance(ht_data_test_prob,"auc"),"y.values")[[1]]
