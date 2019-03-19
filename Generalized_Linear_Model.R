######## Predict 411, Template Code for Unit 2
######## Video of Overview:  https://youtu.be/W4Ub6D0obw8
######## Download appropriate packages and install them from (https://cran.r-project.org/web/packages/available_packages_by_name.html)

# This is generic code that creates a few simple models and does a few simple things with data preparation.

# Some other R Packages to try.
# MICE (and VIM) [for imputation - note they may take a long time to run]
# randomForest [random forest algorithm for classification/regression]
# rpart, rattle, party [decision trees]
# ROCR [allowing you to compute ROC curves, etc.]
# MASS [allows for stepwise variable selection for BIC, AIC, etc.]

# Note, some of these libraries are not needed for this template code.
library(readr)
library(dplyr)
library(zoo)
library(psych)
library(ROCR)
library(corrplot)
library(car)
library(InformationValue)
library(rJava)
library(pbkrtest)
library(car)
library(leaps)
library(MASS)
library(corrplot)
library(glm2)
library(aod)
library(rJava)
library(xlsxjars)
library(xlsx)

# Data Import and Variable Type Changes
setwd("D:/MSDS/411 GLM/Insurance Assignment/Unit 2 Help")
data <- read.csv("logit_insurance.csv")
test <- read.csv("logit_insurance_test.csv")

### Need to make sure our data is understood correctly by R, since we have a mix of numerical and categorical

data$INDEX <- as.factor(data$INDEX)
data$TARGET_FLAG <- as.factor(data$TARGET_FLAG)
data$SEX <- as.factor(data$SEX)
data$EDUCATION <- as.factor(data$EDUCATION)
data$PARENT1 <- as.factor(data$PARENT1)
data$INCOME <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", data$INCOME)))

data$HOME_VAL <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", data$HOME_VAL)))
data$MSTATUS <- as.factor(data$MSTATUS)
data$REVOKED <- as.factor(data$REVOKED)
data$RED_CAR <- as.factor(ifelse(data$RED_CAR=="yes", 1, 0))
data$URBANICITY <- ifelse(data$URBANICITY == "Highly Urban/ Urban", "Urban", "Rural")
data$URBANICITY <- as.factor(data$URBANICITY)
data$JOB <- as.factor(data$JOB)
data$CAR_USE <- as.factor(data$CAR_USE)
data$CAR_TYPE <- as.factor(data$CAR_TYPE)
data$DO_KIDS_DRIVE <- as.factor(ifelse(data$KIDSDRIV > 0, 1, 0 ))
data$OLDCLAIM <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", data$HOME_VAL)))
data$BLUEBOOK <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", data$BLUEBOOK)))
summary(data)


######## Same treatment on test data set ###########################

### Need to make sure our data is understood correctly by R, since we have a mix of numerical and categorical

test$INDEX <- as.factor(test$INDEX)
test$TARGET_FLAG <- as.factor(test$TARGET_FLAG)
test$SEX <- as.factor(test$SEX)
test$EDUCATION <- as.factor(test$EDUCATION)
test$PARENT1 <- as.factor(test$PARENT1)
test$INCOME <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$INCOME)))
test$HOME_VAL <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$HOME_VAL)))
test$MSTATUS <- as.factor(test$MSTATUS)
test$REVOKED <- as.factor(test$REVOKED)
test$RED_CAR <- as.factor(ifelse(test$RED_CAR=="yes", 1, 0))
test$URBANICITY <- ifelse(test$URBANICITY == "Highly Urban/ Urban", "Urban", "Rural")
test$URBANICITY <- as.factor(test$URBANICITY)
test$JOB <- as.factor(test$JOB)
test$CAR_USE <- as.factor(test$CAR_USE)
test$CAR_TYPE <- as.factor(test$CAR_TYPE)
test$DO_KIDS_DRIVE <- as.factor(ifelse(test$KIDSDRIV > 0, 1, 0 ))
test$OLDCLAIM <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$HOME_VAL)))
test$BLUEBOOK <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", test$BLUEBOOK)))
summary(test)

#################### Part 1: Data Exploration ##############################################
# Histograms for Numeric Variables
par(mfrow=c(2,2))
hist(data$AGE, col = "red", xlab = "Age", main = "AGE Hist")
data0<- subset(data, TARGET_FLAG == 1 )
hist(log(data0$TARGET_AMT), col = "green", xlab = "Log TARGET_AMT", main = "Log TARGET_AMT Hist")
boxplot(data$AGE, col = "red", main = "AGE BoxPlot")
boxplot(log(data0$TARGET_AMT), col = "green", main = "LOG TARGET_AMT Boxplot")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(sqrt(data$TRAVTIME), col = "green", xlab = "SQRT TRAVTIME", main = "SQRT TRAVTIME Hist")
hist(data$YOJ, col = "blue", xlab = "YOJ", main = "YOJ Hist")
boxplot(sqrt(data$TRAVTIME), col = "green", main = "SQRT TRAVTIME BoxPlot")
boxplot(data$YOJ, col = "blue", main = "YOJ BoxPlot")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(sqrt(data$BLUEBOOK), col = "green", xlab = "SQRT BLUEBOOK", main = "SQRT BLUEBOOK Hist")
hist((data$TIF), col = "blue", xlab = "TIF", main = "TIF Hist")
boxplot(sqrt(data$BLUEBOOK), col = "green", main = "SQRT BLUEBOOK BoxPlot")
boxplot(data$TIF, col = "blue", main = "TIF BoxPlot")
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(data$MVR_PTS, col = "red", xlab = "MVR_PTS", main = "MVR_PTS Hist")
hist(data$CAR_AGE, col = "blue", xlab = "CAR_AGE", main = "CAR_AGE Hist")
boxplot(data$MVR_PTS, col = "red", main = "MVR_PTS BoxPlot")
boxplot(data$CAR_AGE, col = "blue", xlab = "CAR_AGE", main = "CAR_AGE BoxPlot")
par(mfrow=c(1,1))

########### Part 2: Data Transformation ##################

# Flag Variables
data$INCOME_FLAG <- 0
data$INCOME_FLAG[is.na(data$INCOME)] <- 1
data$OLDCLAIM_FLAG <- 0
data$OLDCLAIM_FLAG[is.na(data$OLDCLAIM)] <- 1
data$YOJ_FLAG <- 0
data$YOJ_FLAG[is.na(data$YOJ)] <- 1
data$HOME_VAL_FLAG <- 0
data$HOME_VAL_FLAG[is.na(data$HOME_VAL)] <- 1
data$HOME_VAL_FLAG_ZERO <- 0
data$HOME_VAL_FLAG_ZERO[data$HOME_VAL == 0] <- 1
data$CAR_AGE_FLAG <- 0
data$CAR_AGE_FLAG[is.na(data$CAR_AGE)] <- 1

# Fix NA's, note car age
data$log_TARGET_AMT <- log(data$TARGET_AMT)
data$AGE[is.na(data$AGE)] <- mean(data$AGE, na.rm = "TRUE")
data$YOJ <- na.aggregate(data$YOJ, data$JOB, mean, na.rm = TRUE)
data$INCOME <- na.aggregate(data$INCOME, data$JOB, mean, na.rm = TRUE)
data$HOME_VAL <- na.aggregate(data$HOME_VAL, data$JOB, mean, na.rm = TRUE )
data$CAR_AGE <- na.aggregate(data$CAR_AGE, data$CAR_TYPE, mean, na.rm = TRUE)
data$CAR_AGE[data$CAR_AGE < 0 ] <- 0 
data$OLDCLAIM <- ifelse(data$CAR_AGE < 5 & !is.na(data$CAR_AGE),0,data$OLDCLAIM)
data$OLDCLAIM <- na.aggregate(data$OLDCLAIM, data$CAR_AGE, mean, na.rm = TRUE )
data$HOME_OWNER <- ifelse(data$HOME_VAL == 0, 0, 1)
data$SQRT_TRAVTIME <- sqrt(data$TRAVTIME)
data$SQRT_BLUEBOOK <- sqrt(data$BLUEBOOK)

# Bin Income
data$INCOME_bin[is.na(data$INCOME)] <- "NA"
data$INCOME_bin[data$INCOME == 0] <- "Zero"
data$INCOME_bin[data$INCOME >= 1 & data$INCOME < 30000] <- "Low"
data$INCOME_bin[data$INCOME >= 30000 & data$INCOME < 80000] <- "Medium"
data$INCOME_bin[data$INCOME >= 80000] <- "High"
data$INCOME_bin <- factor(data$INCOME_bin)
data$INCOME_bin <- factor(data$INCOME_bin, levels=c("NA","Zero","Low","Medium","High"))

# Flag Variabls
test$INCOME_FLAG <- 0
test$INCOME_FLAG[is.na(test$INCOME)] <- 1
test$OLDCLAIM_FLAG <- 0
test$OLDCLAIM_FLAG[is.na(test$OLDCLAIM)] <- 1
test$YOJ_FLAG <- 0
test$YOJ_FLAG[is.na(test$YOJ)] <- 1
test$HOME_VAL_FLAG <- 0
test$HOME_VAL_FLAG[is.na(test$HOME_VAL)] <- 1
test$HOME_VAL_FLAG_ZERO <- 0
test$HOME_VAL_FLAG_ZERO[test$HOME_VAL == 0] <- 1
test$CAR_AGE_FLAG <- 0
test$CAR_AGE_FLAG[is.na(test$CAR_AGE)] <- 1

# Fix NA's
test$AGE[is.na(test$AGE)] <- mean(test$AGE, na.rm = "TRUE")
test$YOJ <- na.aggregate(test$YOJ, test$JOB, mean, na.rm = TRUE)
test$INCOME <- na.aggregate(test$INCOME, test$JOB, mean, na.rm = TRUE)
test$HOME_VAL <- na.aggregate(test$HOME_VAL, test$JOB, mean, na.rm = TRUE)
test$CAR_AGE <- na.aggregate(test$CAR_AGE, test$CAR_TYPE, mean, na.rm = TRUE)
test$CAR_AGE[test$CAR_AGE < 0 ] <- 0 
test$OLDCLAIM <- ifelse(test$CAR_AGE < 5 & !is.na(test$CAR_AGE),0,test$OLDCLAIM)
test$OLDCLAIM <- na.aggregate(test$OLDCLAIM, test$CAR_AGE, mean, na.rm = TRUE )
test$HOME_OWNER <- ifelse(test$HOME_VAL == 0, 0, 1)
test$SQRT_TRAVTIME <- sqrt(test$TRAVTIME)
test$SQRT_BLUEBOOK <- sqrt(test$BLUEBOOK)

# Bin Income
test$INCOME_bin[is.na(test$INCOME)] <- "NA"
test$INCOME_bin[test$INCOME == 0] <- "Zero"
test$INCOME_bin[test$INCOME >= 1 & test$INCOME < 30000] <- "Low"
test$INCOME_bin[test$INCOME >= 30000 & test$INCOME < 80000] <- "Medium"
test$INCOME_bin[test$INCOME >= 80000] <- "High"
test$INCOME_bin <- factor(test$INCOME_bin)
test$INCOME_bin <- factor(test$INCOME_bin, levels=c("NA","Zero","Low","Medium","High"))

summary(data)
summary(test)

numeric <- subset(data, select = c(TARGET_AMT, AGE, HOMEKIDS, YOJ, INCOME, HOME_VAL, TRAVTIME, BLUEBOOK, TIF,
                                   CLM_FREQ, MVR_PTS, CAR_AGE), na.rm = TRUE)
c <- cor(numeric)
corrplot(c, method = "square")

data0<- subset(data, TARGET_FLAG == 1 )
############# Part 3: Model Development ######################
#Model Development for TARGET_FLAG

modelsw <- glm(TARGET_FLAG ~ AGE + PARENT1 + HOME_VAL + 
                 MSTATUS + SEX	+ EDUCATION	+JOB + CAR_USE +	TIF	+ CAR_TYPE + RED_CAR +OLDCLAIM	+ CLM_FREQ + 
                 REVOKED	+ MVR_PTS	+ CAR_AGE + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + 
                 SQRT_BLUEBOOK + INCOME_bin, data = data, family = binomial())
stepwise <- stepAIC(modelsw, direction = "both")
anova(stepwise)
summary(stepwise)
vif(stepwise)

modelsw2 <- glm(TARGET_FLAG ~ AGE + PARENT1 + HOME_VAL + 
                  MSTATUS + SEX	+ EDUCATION	+JOB + CAR_USE +	TIF	+ CAR_TYPE + RED_CAR +OLDCLAIM	+ CLM_FREQ + 
                  REVOKED	+ MVR_PTS	+ CAR_AGE + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + 
                  SQRT_BLUEBOOK + INCOME_bin + INCOME_FLAG + OLDCLAIM_FLAG + YOJ_FLAG +CAR_AGE_FLAG, data = data, family = binomial())
stepwise2 <- stepAIC(modelsw2, direction = "forward")
anova(stepwise2)
summary(stepwise2)
vif(stepwise2)

Model1 <- glm(TARGET_FLAG ~ AGE + BLUEBOOK + TRAVTIME + KIDSDRIV + SEX +  URBANICITY +
                CLM_FREQ + REVOKED + MVR_PTS + TIF + EDUCATION + MSTATUS + PARENT1 + CAR_USE + CAR_TYPE + YOJ + JOB + 
                HOME_VAL,
              data = data, family = binomial())

summary(Model1) 
data$Model1Prediction <- predict(Model1, type = "response")


Model2 <- glm(TARGET_FLAG ~ PARENT1 + MSTATUS + EDUCATION + JOB +CAR_USE + TIF + CAR_TYPE + CLM_FREQ + REVOKED +
                MVR_PTS + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + SQRT_BLUEBOOK + INCOME_bin +
                CAR_AGE_FLAG,
              data = data, family = binomial())		
summary(Model2)
data$Model2Prediction <- predict(Model2, type = "response")

Model3 <- glm(TARGET_FLAG ~ PARENT1 + MSTATUS + EDUCATION + JOB +CAR_USE + TIF + CAR_TYPE + CLM_FREQ + REVOKED +
                MVR_PTS + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + SQRT_BLUEBOOK + INCOME_bin,
              data = data, family = binomial())		

summary(Model3)
data$Model3Prediction <- predict(Model3, type = "response")

###
### Note, we will need to build a quick model for TARGET_AMT.  Perhaps using just a generic multiple linear regression.
###
modelsw3 <- lm(TARGET_AMT ~ AGE + PARENT1 + HOME_VAL + 
                  MSTATUS + SEX	+ EDUCATION	+JOB + CAR_USE +	TIF	+ CAR_TYPE + RED_CAR +OLDCLAIM	+ CLM_FREQ + 
                  REVOKED	+ MVR_PTS	+ CAR_AGE + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + 
                  SQRT_BLUEBOOK + INCOME_bin + INCOME_FLAG + CAR_AGE_FLAG + HOME_VAL_FLAG_ZERO + HOME_VAL_FLAG + YOJ_FLAG + OLDCLAIM_FLAG, data = data0)
stepwise3 <- stepAIC(modelsw3, direction = "both")
summary(stepwise3)

Model1_lm <- lm(TARGET_AMT ~ PARENT1 + MSTATUS + EDUCATION + JOB +CAR_USE + TIF + CAR_TYPE + CLM_FREQ + REVOKED +
                MVR_PTS + URBANICITY + DO_KIDS_DRIVE + HOME_OWNER + SQRT_TRAVTIME + SQRT_BLUEBOOK + INCOME_bin,
              data = data0)																		
summary(Model1_lm)

Model2_lm <- lm(TARGET_AMT ~ MSTATUS + SEX + REVOKED + MVR_PTS + CAR_AGE + HOME_OWNER + SQRT_BLUEBOOK, data = data0)
summary(model2_lm)

########## Part 4: Model Selection 
AIC(Model1)
BIC(Model1)
AIC(Model2)
BIC(Model2)
AIC(Model3)
BIC(Model3)
print(-2*logLik(Model1, REML = TRUE))
print(-2*logLik(Model2, REML = TRUE))
print(-2*logLik(Model3, REML = TRUE))
ks_stat(actuals=data$TARGET_FLAG, predictedScores=data$Model1Prediction)
ks_stat(actuals=data$TARGET_FLAG, predictedScores=data$Model2Prediction)
ks_stat(actuals=data$TARGET_FLAG, predictedScores=data$Model3Prediction)

# We'll choose Model 3, here are its coefficients
coef(Model3)

#### Part 5:  Score Model on Test Data set and output csv file

# Again, double-checking to make sure we don't have any NA's in our Test Data Set
summary(test)

########### STAND ALONE SCORING PROGRAM ###############
##### Model coefficients used to create P_TARGET_FLAG, Mean TARGET_AMT aggregated by CAR was used to predict
######## P-TARGET AMOUNT
######## Note that you may want to do a multiple linear regression for this model.
######## The model below is based on CAR multiplied by the amount and probability of a claim (27%)
test$P_TARGET_FLAG <- predict(Model3, newdata = test, type = "response")
test$P_TARGET_AMT_INITIAL <- predict(Model2_lm, newdata = test, type = "response")
test$P_TARGET_AMT <- test$P_TARGET_FLAG * test$P_TARGET_AMT_INITIAL

summary(test$P_TARGET_AMT_INITIAL)
summary(test$P_TARGET_AMT)

# Scored Data File
scores <- test[c("INDEX","P_TARGET_FLAG", "P_TARGET_AMT")]
write.xlsx(scores, file = "logit_insurance_test.xlsx", sheetName = "Scored Data File", row.names = F)

