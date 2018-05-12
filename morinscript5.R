################################################################
##### Big Data HW 5 ####################################
##### By Blain Morin ##########################
############################################

require(readr) || install.packages("readr")
require(dplyr) || install.packages("dplyr")
require(caret) || install.packages("caret", dependencies = TRUE)
require(doParallel) || install.packages("doParallel")
require(gbm) || install.packages("gbm", dependencies = TRUE)
require(nnet) || install.packages("nnet", dependencies = TRUE)
require(mgcv) || installed.packages("mgcv", dependencies = TRUE)

authors = function() {
  c("Blain Morin")
}


set.seed(100)

### Read in training, private test, and public test data
Predict_NoShow_Train = read_csv("Predict_NoShow_Train.csv")
Predict_NoShow_PublicTest_WithoutLabels = read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv")
Predict_NoShow_PrivateTest_WithoutLabels = read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv")


### Identify which columns are factors
factors = c("Gender",
            "Diabetes",
            "Alcoholism",
            "Hypertension",
            "Handicapped",
            "Smoker",
            "Scholarship",
            "Tuberculosis",
            "RemindedViaSMS",
            "DayOfTheWeek",
            "Status")


##################################################
### Clean Test Data ##############################
#################################################


### First factor Status so that No-Shows are = 1
Predict_NoShow_Train$Status = ifelse(Predict_NoShow_Train$Status == "No-Show", 1, 0)

### Trim the low count categories (improves fit)
Predict_NoShow_Train$Handicapped = ifelse(Predict_NoShow_Train$Handicapped >= 1, 1, 0)
Predict_NoShow_Train$RemindedViaSMS = ifelse(Predict_NoShow_Train$RemindedViaSMS >= 1, 1, 0)
Predict_NoShow_Train[factors] = as.data.frame(lapply(Predict_NoShow_Train[factors], factor))

### Remove non predictors Dates and ID
### Remove predictors with low influence (Tuberculosis, Day of the Week, Hypertension, Diabetes)
Predict_NoShow_Train = Predict_NoShow_Train %>% select(-c(DateAppointmentWasMade,
                                                         DateOfAppointment, ID, Tuberculosis, DayOfTheWeek, Hypertension, Diabetes))

### Center and scale continuous predictors (improves fit)
Predict_NoShow_Train$Age = scale(Predict_NoShow_Train$Age)
Predict_NoShow_Train$DaysUntilAppointment = scale(Predict_NoShow_Train$DaysUntilAppointment)

### Create model matrix
test.preds = model.matrix(Status ~ . , data = Predict_NoShow_Train)
### Remove intercept column
test.preds = test.preds[,-1]

### Create outcome vector
test.out = Predict_NoShow_Train$Status


################################################
### Clean Public Data ##########################
###############################################


### Trim low count categories (to match training data)
Predict_NoShow_PublicTest_WithoutLabels$Handicapped = ifelse(Predict_NoShow_PublicTest_WithoutLabels$Handicapped >= 1, 1, 0)
Predict_NoShow_PublicTest_WithoutLabels$RemindedViaSMS = ifelse(Predict_NoShow_PublicTest_WithoutLabels$RemindedViaSMS >= 1, 1, 0)
Predict_NoShow_PublicTest_WithoutLabels = Predict_NoShow_PublicTest_WithoutLabels %>% 
  select(-c(DateAppointmentWasMade, DateOfAppointment))
Predict_NoShow_PublicTest_WithoutLabels[factors[-11]] = 
  as.data.frame(lapply(Predict_NoShow_PublicTest_WithoutLabels[factors[-11]], factor))

### Center and Scale
Predict_NoShow_PublicTest_WithoutLabels$Age = scale(Predict_NoShow_PublicTest_WithoutLabels$Age)
Predict_NoShow_PublicTest_WithoutLabels$DaysUntilAppointment = scale(Predict_NoShow_PublicTest_WithoutLabels$DaysUntilAppointment)

### Create model matrix
pub.preds = model.matrix(~ . , data = Predict_NoShow_PublicTest_WithoutLabels)
pub.preds = pub.preds[,-1]


############################################
### Clean Private Data #####################
###########################################


### Trim small and unused categories
Predict_NoShow_PrivateTest_WithoutLabels$Handicapped = ifelse(Predict_NoShow_PrivateTest_WithoutLabels$Handicapped >= 1, 1, 0)
Predict_NoShow_PrivateTest_WithoutLabels$RemindedViaSMS = ifelse(Predict_NoShow_PrivateTest_WithoutLabels$RemindedViaSMS >= 1, 1, 0)
Predict_NoShow_PrivateTest_WithoutLabels = Predict_NoShow_PrivateTest_WithoutLabels %>%
  select(-c(DateAppointmentWasMade,
            DateOfAppointment))
Predict_NoShow_PrivateTest_WithoutLabels[factors[-11]] = 
  as.data.frame(lapply(Predict_NoShow_PrivateTest_WithoutLabels[factors[-11]], factor))

### Center and scale
Predict_NoShow_PrivateTest_WithoutLabels$Age = scale(Predict_NoShow_PrivateTest_WithoutLabels$Age)
Predict_NoShow_PrivateTest_WithoutLabels$DaysUntilAppointment = scale(Predict_NoShow_PrivateTest_WithoutLabels$DaysUntilAppointment)

### Create model matrix
priv.preds = model.matrix(~ . , data = Predict_NoShow_PrivateTest_WithoutLabels)
priv.preds = priv.preds[,-1]


########################################
### Tune GBM Model ####################
######################################

### Give levels character names (necessary for caret to run)
levels(test.out) = c("show", "noshow")

### Initialize parallelization
cl = makeCluster(detectCores())
registerDoParallel(cl)


### Set training paramters
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

### Create a list of parameters to check
gbmgrid = expand.grid(interaction.depth = 1:8, 
                       n.trees = seq(50, 400, by = 50),
                       shrinkage = seq(0, .5, by = .1),
                       n.minobsinnode = seq(1 , 16, by = 5))

### Train the gbm model
gmbfit = train(x = test.preds, 
               y = test.out, 
               method = "gbm", 
               trControl=objControl, 
               verbose = F,
               tuneGrid = gbmgrid)



stopCluster(cl)




### Predict on test sets
preds = predict.train(gmbfit, newdata = pub.preds, type = "prob")
preds2 = predict.train(gmbfit, newdata = priv.preds, type = "prob")

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, preds$noshow))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, preds2$noshow))


### Write csv
write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)




#####################################################
### Tune NN #########################################
####################################################

### Give levels character names (necessary for caret to run)
levels(test.out) = c("show", "noshow")

### Initialize parallization
cl = makeCluster(detectCores())
registerDoParallel(cl)

### Set training parameters
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

### Make list of tuning parameters to check in the NN model
nngrid = expand.grid(size = 2:12, 
                      decay = seq(0, 1, by = .1))


### Train the model
nnfit = train(x = test.preds, 
               y = test.out, 
               method = "nnet", 
               trControl=objControl,
              tuneGrid = nngrid, maxit = 1000)



stopCluster(cl)


### Predict on test sets
nnpreds = predict.train(nnfit, newdata = pub.preds, type = "prob")
nnpreds2 = predict.train(nnfit, newdata = priv.preds, type = "prob")

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, nnpreds$noshow))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, nnpreds2$noshow))


### Write csv
write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)



##################################################
### Tune GAM ###################################
##############################################

### Give levels character names (necessary for caret to run)
levels(test.out) = c("show", "noshow")

### Initialize parallization
cl = makeCluster(detectCores())
registerDoParallel(cl)

### Set training parameters
objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

### Make list of tuning parameters to check in the NN model
gamgrid = expand.grid(size = 2:12, 
                     decay = seq(0, 1, by = .1))


### Train the model
gamfit = train(x = test.preds, 
              y = test.out, 
              method = "gam", 
              trControl=objControl)



stopCluster(cl)

### Predict on test sets
gampreds = predict.train(gamfit, newdata = pub.preds, type = "prob")
gampreds2 = predict.train(gamfit, newdata = priv.preds, type = "prob")

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, gampreds$noshow))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, gampreds2$noshow))


### Write csv
write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)

print(gamfit)

##################################################
### Logistic Regression #######################
##############################################


### Run a basic logistic regression 
logit = glm(data = Predict_NoShow_Train,
            Status ~ Age + Gender + Diabetes + Alcoholism + Hypertension +
              Handicapped + Smoker + Scholarship + Tuberculosis + RemindedViaSMS +
              DayOfTheWeek + DaysUntilAppointment, family = binomial(link = "logit"))


### Predict on the test sets
logit.preds = predict(logit, newdata = Predict_NoShow_PublicTest_WithoutLabels)
logit.preds2 = predict(logit, newdata = Predict_NoShow_PrivateTest_WithoutLabels)

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, logit.preds))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, logit.preds2))


### Write csv
write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)
