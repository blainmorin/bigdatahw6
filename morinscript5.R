################################################################
##### Big Data HW 5 ####################################
##### By Blain Morin ##########################
############################################

require(readr) || install.packages("readr")
require(dplyr) || install.packages("dplyr")
require(caret) || install.packages("caret", dependencies = TRUE)
require(doParallel) || install.packages("doParallel")

authors = function() {
  c("Blain Morin")
}


set.seed(100)

Predict_NoShow_Train = read_csv("Predict_NoShow_Train.csv")
Predict_NoShow_PublicTest_WithoutLabels = read_csv("Predict_NoShow_PublicTest_WithoutLabels.csv")
Predict_NoShow_PrivateTest_WithoutLabels = read_csv("Predict_NoShow_PrivateTest_WithoutLabels.csv")

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

### Trim the low count categories (improves gbm predictions)
Predict_NoShow_Train$Handicapped = ifelse(Predict_NoShow_Train$Handicapped >= 1, 1, 0)
Predict_NoShow_Train$RemindedViaSMS = ifelse(Predict_NoShow_Train$RemindedViaSMS >= 1, 1, 0)
Predict_NoShow_Train[factors] = as.data.frame(lapply(Predict_NoShow_Train[factors], factor))
Predict_NoShow_Train = Predict_NoShow_Train %>% select(-c(DateAppointmentWasMade,
                                                         DateOfAppointment, ID, Tuberculosis, DayOfTheWeek, Hypertension, Diabetes))

### Center and scale continuous predictors (improves gbm predictions)
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

## Set up parallelization 

levels(test.out) = c("show", "noshow")

cl = makeCluster(detectCores())
registerDoParallel(cl)

objControl <- trainControl(method='cv', number=5, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

gbmgrid = expand.grid(interaction.depth = 1:8, 
                       n.trees = seq(50, 400, by = 50),
                       shrinkage = seq(0, .5, by = .1),
                       n.minobsinnode = seq(1 , 16, by = 5))

gmbfit = train(x = test.preds, 
               y = test.out, 
               method = "gbm", 
               trControl=objControl, 
               verbose = F,
               tuneGrid = gbmgrid)



stopCluster(cl)





preds = predict.train(gmbfit, newdata = pub.preds, type = "prob")
preds2 = predict.train(gmbfit, newdata = priv.preds, type = "prob")

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, preds$noshow))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, preds2$noshow))

write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)




####If no finish
tester = glm(data = Predict_NoShow_Train,
            Status ~ Age + Gender + Diabetes + Alcoholism + Hypertension +
              Handicapped + Smoker + Scholarship + Tuberculosis + RemindedViaSMS +
              DayOfTheWeek + DaysUntilAppointment, family = binomial(link = "logit"))

predict(tester)

preds = predict(tester, newdata = Predict_NoShow_PublicTest_WithoutLabels)
preds2 = predict(tester, newdata = Predict_NoShow_PrivateTest_WithoutLabels)

public = as.data.frame(cbind(Predict_NoShow_PublicTest_WithoutLabels$ID, preds))
private = as.data.frame(cbind(Predict_NoShow_PrivateTest_WithoutLabels$ID, preds2))

write.table(public, file = "public.csv", sep = ",", col.names = FALSE, row.names = FALSE)
write.table(private, file = "private.csv", sep = ",", col.names = FALSE, row.names = FALSE)
