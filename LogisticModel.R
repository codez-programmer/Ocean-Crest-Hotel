library(tidyverse)
library(ggplot2)
library(skimr)
library(dplyr)
library(GGally)
library(caret)
library(car)

# Exploratory Data Analysis
hotel_data<- read.csv("D://Direct Downloads//PredictiveAnalytics//hotel_data.csv", header= T)
hotel_data$X <- NULL
hotel_data$country <- NULL
hotel_data$cases_info <- NULL
head(hotel_data)
tail(hotel_data)


# Predictive Analysis
summaryStats <- skim(hotel_data)
summaryStats

hotel_data <- hotel_data %>% mutate_at(c("is_canceled","is_repeated_guest","hotel","deposit_type","customer_type","reserved_room_type","assigned_room_type","meal","market_segment","distribution_channel"), as.factor) 

#2. rename resonse 
hotel_data$is_canceled<-fct_recode(hotel_data$is_canceled, cancel = "1",notcancel = "0")

#3. relevel response
hotel_data$is_canceled<- relevel(hotel_data$is_canceled, ref = "cancel")

#make sure levels are correct
levels(hotel_data$is_canceled)

hotel_data_dummy <- model.matrix(is_canceled~ ., data = hotel_data)#create dummy variables expect for the response
hotel_data_dummy<- data.frame(hotel_data_dummy[,-1]) #get rid of intercept
hotel_data <- cbind(is_canceled=hotel_data$is_canceled, hotel_data_dummy)


set.seed(99) #set random seed
index <- createDataPartition(hotel_data$is_canceled, p = .8,list = FALSE)
hotel_data_train <-hotel_data[index,]
hotel_data_test <- hotel_data[-index,]

library(e1071)
library(glmnet)
library(Matrix)
library(doParallel)

#total number of cores on your computer
num_cores<-detectCores(logical=FALSE)
num_cores

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)


# Variable Selection Forward 
set.seed(10)#set the seed again since within the train method the validation set is randomly selected
hotel_data_model_forward <- train(is_canceled ~ .,
                                        data = hotel_data_train,
                                        method = "glmStepAIC",
                                        direction="forward",
                                        trControl =trainControl(method = "none",
                                                                classProbs = TRUE,
                                                                summaryFunction = twoClassSummary),
                                        metric="ROC")

#stop parallel processing
stopCluster(cl)
registerDoSEQ()

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

# Variable Selection Backward 
set.seed(10)#set the seed again since within the train method the validation set is randomly selected
hotel_data_model_backward <- train(is_canceled ~ .,
                                         data = hotel_data_train,
                                         method = "glmStepAIC",
                                         direction="backward",
                                         trControl =trainControl(method = "none",
                                                                 classProbs = TRUE,
                                                                 summaryFunction = twoClassSummary),
                                         metric="ROC")
#stop parallel processing
stopCluster(cl)
registerDoSEQ()

#start parallel processing
cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)

# Lasso Model
set.seed(10)#set the seed again since within the train method the validation set is randomly selected
hotel_data_model_lasso <- train(is_canceled ~ .,
                                      data = hotel_data_train,
                                      method = "glmnet",
                                      standardize =T,
                                      tuneGrid = expand.grid(alpha =1, #lasso
                                                             lambda = seq(0.0001, 1, length = 20)),
                                      trControl =trainControl(method = "cv",
                                                              number = 5,
                                                              classProbs = TRUE,
                                                              summaryFunction = twoClassSummary),
                                      metric="ROC")

#stop parallel processing
stopCluster(cl)
registerDoSEQ()

#First, get the predicted probabilities of the test data.
predprob_hotel_data_model_forward<-predict(hotel_data_model_forward , hotel_data_test, type="prob")
predprob_hotel_data_model_backward<-predict(hotel_data_model_backward , hotel_data_test, type="prob")
predprob_hotel_data_model_lasso<-predict(hotel_data_model_lasso , hotel_data_test, type="prob")


install.packages("ROCR")
library(ROCR)
pred_forward <- prediction(predprob_hotel_data_model_forward$cancel, hotel_data_test$is_canceled,label.ordering =c("notcancel","cancel") )
pred_backward<- prediction(predprob_hotel_data_model_backward$cancel, hotel_data_test$is_canceled,label.ordering =c("notcancel","cancel") )
pred_lasso <- prediction(predprob_hotel_data_model_lasso$cancel, hotel_data_test$is_canceled,label.ordering =c("notcancel","cancel") )

#Get the AUC
auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))
auc_lasso

auc_forward<-unlist(slot(performance(pred_forward, "auc"), "y.values"))
auc_forward

auc_backward<-unlist(slot(performance(pred_backward, "auc"), "y.values"))
auc_backward

t<- performance(pred_forward,"prec", "rec")

perf1 <- performance(pred_forward, "prec", "rec")
plot(perf1)
