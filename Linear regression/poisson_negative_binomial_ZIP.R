######## Modified from Template Code from NU 411
######## Video of Overview:  https://youtu.be/ZUoyztUSjfI
######## Download appropriate packages and install them from (https://cran.r-project.org/web/packages/available_packages_by_name.html)

# Note that some of the Zero Inflated models will take a few seconds/moments to run.  Especially if you have a larger
# number of variables involved in the model.

#Read File in from your working directory
setwd("C:/Desktop/MSDS/Winter MSDS/411 MLR/Wine Sales")
wine = read.csv("WINE_Training.csv")  # read csv file

#call libraries
library(ggplot2) # For graphical tools
library(MASS) # For some advanced statistics
library(pscl) # For "counting" models (e.g., Poisson and Negative Binomial)
library(dplyr) # For general needs and functions
library(readr)
library(corrplot)
library("mice")
library(randomForest)

# Note, some of these libraries are not needed for this template code.
library(zoo)
library(psych)
library(ROCR)
library(car)
library(InformationValue)
library(rJava)
library(pbkrtest)
library(car)
library(leaps)
library(glm2)
library(aod)

#take a look at the high level characteristics of the wine data
summary(wine)
str(wine)

#examine the target variable
ggplot(data=wine, aes(wine$TARGET)) + 
  geom_histogram(binwidth =1, 
                 col="BLUE", 
                 aes(fill=..count..))+
  scale_fill_gradient("Count", low = "blue", high = "red")

wine_clean <- na.omit(wine)
cor(wine_clean[sapply(wine_clean, is.numeric)])

wine_corr <- cor(wine_clean[,2:10])
corrplot(wine_corr, method = "circle")

wine_corr <- cor(wine_clean[,c(2,11:16)])
corrplot(wine_corr, method = "circle")

# Replace negative values
par(mar=c(4,7,2,1)) 
hist(wine$ResidualSugar, col = "dodgerblue2", main = "Hist of ResidualSugar")
wine[which(wine$Alcohol < quantile(wine$Alcohol,.01,na.rm=T)),"Alcohol"] = quantile(wine$Alcohol,.01,na.rm=T)
wine[which(wine$Alcohol < quantile(wine$Alcohol,.99,na.rm=T)),"Alcohol"] = quantile(wine$Alcohol,.99,na.rm=T)
wine[which(wine$Chlorides < quantile(wine$Chlorides,.3,na.rm=T)),"Chlorides"] =  quantile(wine$Chlorides,.3,na.rm=T)
wine[which(wine$Chlorides < quantile(wine$Chlorides,.7,na.rm=T)),"Chlorides"] =  quantile(wine$Chlorides,.7,na.rm=T)
wine[which(wine$CitricAcid < quantile(wine$CitricAcid,.25,na.rm=T)),"CitricAcid"] =  quantile(wine$CitricAcid,.25,na.rm=T)
wine[which(wine$CitricAcid < quantile(wine$CitricAcid,.75,na.rm=T)),"CitricAcid"] =  quantile(wine$CitricAcid,.75,na.rm=T)
wine[which(wine$FixedAcidity < quantile(wine$FixedAcidity,.13,na.rm=T)),"FixedAcidity"] =  quantile(wine$FixedAcidity,.13,na.rm=T)
wine[which(wine$FixedAcidity < quantile(wine$FixedAcidity,.87,na.rm=T)),"FixedAcidity"] =  quantile(wine$FixedAcidity,.87,na.rm=T)
wine[which(wine$FreeSulfurDioxide < quantile(wine$FreeSulfurDioxide,.25,na.rm=T)),"FreeSulfurDioxide"] =  quantile(wine$FreeSulfurDioxide,.25,na.rm=T)
wine[which(wine$FreeSulfurDioxide < quantile(wine$FreeSulfurDioxide,.75,na.rm=T)),"FreeSulfurDioxide"] =  quantile(wine$FreeSulfurDioxide,.75,na.rm=T)
wine[which(wine$ResidualSugar < quantile(wine$ResidualSugar,.3,na.rm=T)),"ResidualSugar"] =  quantile(wine$ResidualSugar,.3,na.rm=T)
wine[which(wine$ResidualSugar < quantile(wine$ResidualSugar,.7,na.rm=T)),"ResidualSugar"] =  quantile(wine$ResidualSugar,.7,na.rm=T)
wine[which(wine$Sulphates < quantile(wine$Sulphates,.22,na.rm=T)),"Sulphates"] =  quantile(wine$Sulphates,.22,na.rm=T)
wine[which(wine$Sulphates < quantile(wine$Sulphates,.78,na.rm=T)),"Sulphates"] =  quantile(wine$Sulphates,.78,na.rm=T)
wine[which(wine$TotalSulfurDioxide < quantile(wine$TotalSulfurDioxide,.21,na.rm=T)),"TotalSulfurDioxide"] =  quantile(wine$TotalSulfurDioxide,.21,na.rm=T)
wine[which(wine$TotalSulfurDioxide < quantile(wine$TotalSulfurDioxide,.79,na.rm=T)),"TotalSulfurDioxide"] =  quantile(wine$TotalSulfurDioxide,.79,na.rm=T)
wine[which(wine$VolatileAcidity < quantile(wine$VolatileAcidity,.23,na.rm=T)),"VolatileAcidity"] =  quantile(wine$VolatileAcidity,.23,na.rm=T)
wine[which(wine$VolatileAcidity < quantile(wine$VolatileAcidity,.77,na.rm=T)),"VolatileAcidity"] =  quantile(wine$VolatileAcidity,.77,na.rm=T)
wine[which(wine$AcidIndex < quantile(wine$AcidIndex,.01,na.rm=T)),"AcidIndex"] = quantile(wine$AcidIndex,.01,na.rm=T)
wine[which(wine$AcidIndex > quantile(wine$AcidIndex,.99,na.rm=T)),"AcidIndex"] = quantile(wine$AcidIndex,.99,na.rm=T)

#create IMP versions of each independent variable
wine$FixedAcidity_IMP <- wine$FixedAcidity
wine$VolatileAcidity_IMP <- wine$VolatileAcidity
wine$CitricAcid_IMP <- wine$CitricAcid
wine$ResidualSugar_IMP <- wine$ResidualSugar
wine$Chlorides_IMP <- wine$Chlorides
wine$FreeSulfurDioxide_IMP <- wine$FreeSulfurDioxide
wine$TotalSulfurDioxide_IMP <- wine$TotalSulfurDioxide
wine$Density_IMP <- wine$Density
wine$pH_IMP <- wine$pH
wine$Sulphates_IMP <- wine$Sulphates
wine$Alcohol_IMP <- wine$Alcohol
wine$LabelAppeal_IMP <- wine$LabelAppeal
wine$AcidIndex_IMP <- wine$AcidIndex
wine$STARS_IMP <- wine$STARS

wine.multiple <- wine #copy for multiple imputation
wine.multiple <- wine.multiple[,c(1,2, 17:30)]


# Create a flag for missingness for each column
for(i in 2:length(names(wine))){
  nm.orig<-names(wine)[i]
  nm<-paste(names(wine)[i],"_FLAG",sep="")
  wine[,nm]<-ifelse(is.na(wine[,nm.orig])==TRUE,1,0)
  
}

for(i in 2:length(names(wine.multiple))){
  nm.orig<-names(wine.multiple)[i]
  nm<-paste(names(wine.multiple)[i],"_FLAG",sep="")
  wine.multiple[,nm]<-ifelse(is.na(wine.multiple[,nm.orig])==TRUE,1,0)
}

# impute missing values 
wine.impute <- wine.multiple[,c(1:16)]
# Don't need code below as all variables are numeric, but for future reference
# as.factor may be needed before imputation.
#library(dplyr) 
#wine.multiple <- wine.multiple %>%
# mutate(
#   Chlorides = as.numerc(Chlorides),
#   STARS = as.numeric(STARS),
#   FixedAcidity = as.numeric(FixedAcidity)
# )

init = mice(wine.impute, maxit=0) 
meth = init$method
predM = init$predictorMatrix
# The code below will remove the variable as predictor but still will be imputed. 
predM[,c("INDEX")]=0

# To skip a variable from imputation use the code below. This variable will be used for prediction.
meth[c("FixedAcidity_IMP")]=""
meth[c("VolatileAcidity_IMP")]=""
meth[c("CitricAcid_IMP")]=""
meth[c("Density_IMP")]=""
meth[c("LabelAppeal_IMP")]=""
meth[c("AcidIndex_IMP")]=""

meth
meth[c("ResidualSugar_IMP","Chlorides_IMP","FreeSulfurDioxide_IMP","TotalSulfurDioxide_IMP", "pH_IMP", "Sulphates_IMP",
       "Alcohol_IMP", "STARS_IMP")]="norm" 
set.seed(103)
imputed = mice(wine.impute, method=meth, predictorMatrix=predM, m=5)
imputed <- complete(imputed)
sapply(imputed, function(x) sum(is.na(x)))

summary(imputed)
summary(wine)

wine.multiple <- wine.multiple[,c(1,21:24, 26:28, 31)] #Col 17 dropped as its target value
# All flag columns that didn't have nas are dropped

imputed <- merge(x = imputed, y = wine.multiple, by = "INDEX", all.x = TRUE)

#replace NA's in each column with mean
wine$FixedAcidity_IMP[which(is.na(wine$FixedAcidity_IMP))] <- mean(wine$FixedAcidity_IMP,na.rm = TRUE)
wine$VolatileAcidity_IMP[which(is.na(wine$VolatileAcidity_IMP))] <- mean(wine$VolatileAcidity_IMP,na.rm = TRUE)
wine$CitricAcid_IMP[which(is.na(wine$CitricAcid_IMP))] <- mean(wine$CitricAcid_IMP,na.rm = TRUE)
wine$ResidualSugar_IMP[which(is.na(wine$ResidualSugar_IMP))] <- mean(wine$ResidualSugar_IMP,na.rm = TRUE)
wine$Chlorides_IMP[which(is.na(wine$Chlorides_IMP))] <- mean(wine$Chlorides_IMP,na.rm = TRUE)
wine$FreeSulfurDioxide_IMP[which(is.na(wine$FreeSulfurDioxide_IMP))] <- mean(wine$FreeSulfurDioxide_IMP,na.rm = TRUE)
wine$TotalSulfurDioxide_IMP[which(is.na(wine$TotalSulfurDioxide_IMP))] <- mean(wine$TotalSulfurDioxide_IMP,na.rm = TRUE)
wine$Density_IMP[which(is.na(wine$Density_IMP))] <- mean(wine$Density_IMP,na.rm = TRUE)
wine$pH_IMP[which(is.na(wine$pH_IMP))] <- mean(wine$pH_IMP,na.rm = TRUE)
wine$Sulphates_IMP[which(is.na(wine$Sulphates_IMP))] <- mean(wine$Sulphates_IMP,na.rm = TRUE)
wine$Alcohol_IMP[which(is.na(wine$Alcohol_IMP))] <- mean(wine$Alcohol_IMP,na.rm = TRUE)
wine$STARS_IMP[which(is.na(wine$STARS_IMP))] <- mean(wine$STARS_IMP,na.rm = TRUE)

#Is it possible to distinguish red vs white wines by the chemical property makeup?
plot(wine$VolatileAcidity_IMP)

#A better way to visualize volatile acidity
ggplot(data=wine, aes(wine$VolatileAcidity_IMP)) + 
  geom_histogram(binwidth =1, 
                 col="BLUE", 
                 aes(fill=..count..))+
  scale_fill_gradient("Count", low = "blue", high = "red")

summary(wine$VolatileAcidity_IMP)

#make new indicator that indicates red vs white based on volatile acidity
wine$VolatileAcidity_IMP_REDFLAG <- ifelse(wine$VolatileAcidity_IMP > mean(wine$VolatileAcidity_IMP),1,0)
wine$ResidualSugar_IMP_REDFLAG <- ifelse(wine$ResidualSugar_IMP < mean(wine$ResidualSugar_IMP),1,0)
wine$TotalSulfurDioxide_IMP_REDFLAG <- ifelse(wine$TotalSulfurDioxide_IMP < mean(wine$TotalSulfurDioxide_IMP),1,0)
wine$Density_IMP_REDFLAG <- ifelse(wine$Density_IMP > mean(wine$Density_IMP),1,0)
wine$TallyUp <- wine$VolatileAcidity_IMP_REDFLAG + wine$ResidualSugar_IMP_REDFLAG + wine$TotalSulfurDioxide_IMP_REDFLAG + wine$Density_IMP_REDFLAG
wine$Final_REDFLAG <- ifelse(wine$TallyUp > mean(wine$TallyUp),1,0)

imputed$VolatileAcidity_IMP_REDFLAG <- ifelse(imputed$VolatileAcidity_IMP > mean(imputed$VolatileAcidity_IMP),1,0)
imputed$ResidualSugar_IMP_REDFLAG <- ifelse(imputed$ResidualSugar_IMP < mean(imputed$ResidualSugar_IMP),1,0)
imputed$TotalSulfurDioxide_IMP_REDFLAG <- ifelse(imputed$TotalSulfurDioxide_IMP < mean(imputed$TotalSulfurDioxide_IMP),1,0)
imputed$Density_IMP_REDFLAG <- ifelse(imputed$Density_IMP > mean(imputed$Density_IMP),1,0)
imputed$TallyUp <- imputed$VolatileAcidity_IMP_REDFLAG + imputed$ResidualSugar_IMP_REDFLAG + imputed$TotalSulfurDioxide_IMP_REDFLAG + imputed$Density_IMP_REDFLAG
imputed$Final_REDFLAG <- ifelse(imputed$TallyUp > mean(imputed$TallyUp),1,0)


pairs(wine[,c("Final_REDFLAG","VolatileAcidity_IMP")])

plot( wine$VolatileAcidity_IMP,wine$TARGET)

#Add Target Flag for 0 sale scenarios
wine$TARGET_Flag <- ifelse(wine$TARGET >0,1,0)
wine$TARGET_AMT <- wine$TARGET - 1
wine$TARGET_AMT <- ifelse(wine$TARGET_Flag == 0,NA,wine$TARGET-1)

imputed$TARGET_Flag <- ifelse(imputed$TARGET >0,1,0)
imputed$TARGET_AMT <- imputed$TARGET - 1
imputed$TARGET_AMT <- ifelse(imputed$TARGET_Flag == 0,NA,imputed$TARGET-1)

#######################################################
################# Model Building ##################
#Function for Mean Square Error Calculation
mse <- function(sm) 
  mean(sm$residuals^2)

R2 <- function(fitted, true){
  1 - (sum((true - fitted)^2)/sum((true - mean(true))^2))
}

## FIRST MODEL ... REGULAR LINEAR REGRESSION MODEL#####
lm_fit <- lm(TARGET~ STARS_IMP + LabelAppeal_IMP + AcidIndex_IMP
             + Alcohol_IMP + Chlorides_IMP + CitricAcid_IMP + Density_IMP + FixedAcidity_IMP
             + FreeSulfurDioxide_IMP + LabelAppeal_IMP + ResidualSugar_IMP + STARS_IMP + Sulphates_IMP
             + TotalSulfurDioxide_IMP + VolatileAcidity_IMP + pH_IMP, data = wine)

summary(lm_fit)
coefficients(lm_fit)
wine$fittedLM <-fitted(lm_fit)
AIC(lm_fit)
mse(lm_fit)
lm_fit.Adj_R2.test <- summary(lm_fit)$adj.r.squared

# Using imputed dataset instead.
lm_fit <- lm(TARGET~ STARS_IMP + LabelAppeal_IMP + AcidIndex_IMP
             + Alcohol_IMP + Chlorides_IMP + CitricAcid_IMP + Density_IMP + FixedAcidity_IMP
             + FreeSulfurDioxide_IMP + LabelAppeal_IMP + ResidualSugar_IMP + STARS_IMP + Sulphates_IMP
             + TotalSulfurDioxide_IMP + VolatileAcidity_IMP + pH_IMP, data = imputed)

summary(lm_fit)
coefficients(lm_fit)
wine$fittedLM <-fitted(lm_fit)
AIC(lm_fit)
mse(lm_fit)
lm_fit.Adj_R2.test <- summary(lm_fit)$adj.r.squared

lm_fit <- lm(TARGET~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP 
             + AcidIndex_IMP + ResidualSugar_IMP 
             + Density_IMP_REDFLAG + ResidualSugar_IMP_REDFLAG + TotalSulfurDioxide_IMP_REDFLAG
             + TallyUp + Final_REDFLAG, data = imputed)

summary(lm_fit)
coefficients(lm_fit)
imputed$fittedLM <-fitted(lm_fit)
AIC(lm_fit)
mse(lm_fit)
lm_fit.Adj_R2.test <- summary(lm_fit)$adj.r.squared

##########################################################################################
##########################################################################################
## SECOND MODEL ... REGULAR LINEAR REGRESSION MODEL USING STEPWISE VARIABLE SELECTION (AIC)
##########################################################################################

stepwise_lm <- stepAIC(lm_fit, direction="both")
stepwise_lm$anova


lm_fit_stepwise <- lm(TARGET ~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP + AcidIndex_IMP + 
                        ResidualSugar_IMP_REDFLAG + Density_IMP_REDFLAG + TallyUp, data=imputed)
summary(lm_fit_stepwise)
coefficients(lm_fit_stepwise)
imputed$fittedLMStepwise <-fitted(lm_fit_stepwise)
AIC(lm_fit_stepwise)
mse(lm_fit_stepwise)
lm_fit_stepwise.Adj_R2.test <- summary(lm_fit_stepwise)$adj.r.squared


##########################################################################################
##########################################################################################
## THIRD MODEL ... POISSON################################################################
##########################################################################################

poisson_model <- glm(TARGET ~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP 
                     + AcidIndex_IMP + ResidualSugar_IMP_REDFLAG + Density_IMP_REDFLAG 
                     + TallyUp, family="poisson"(link="log"), data=imputed)

summary(poisson_model)
AIC(poisson_model)
mse(poisson_model)

coef(poisson_model)

imputed$poisson_yhat <- predict(poisson_model, newdata = imputed, type = "response")
poisson_model.test <- imputed$poisson_yhat
poisson_model.R2 <- R2(poisson_model.test, imputed$TARGET)
poisson_model.Adj_R2.test <- 1 - (1 - poisson_model.R2) * (nrow(imputed) - 1)/(nrow(imputed)-(length(poisson_model$coefficients)-1)-1)


##########################################################################################
##########################################################################################
## FOURTH MODEL ... NEGATIVE BINOMIAL DISTRIBUTION########################################
##########################################################################################

NBR_Model<-glm.nb(TARGET ~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP + AcidIndex_IMP + 
                    ResidualSugar_IMP_REDFLAG + Density_IMP_REDFLAG + TallyUp, data=imputed)

summary(NBR_Model)
AIC(NBR_Model)
mse(NBR_Model)

imputed$NBRphat <- predict(NBR_Model, newdata = imputed, type = "response")
NBR_Model.test <- imputed$NBRphat
NBR_Model.R2 <- R2(NBR_Model.test, imputed$TARGET)
NBR_Model.Adj_R2.test <- 1 - (1 - NBR_Model.R2) * (nrow(imputed) - 1)/(nrow(imputed)-(length(NBR_Model$coefficients)-1)-1)

##########################################################################################
##########################################################################################
## FIFTH MODEL ... ZERO INFLATED POISSON (ZIP)############################################
##########################################################################################

ZIP_Model<-zeroinfl(TARGET ~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP + AcidIndex_IMP + 
                      ResidualSugar_IMP_REDFLAG + Density_IMP_REDFLAG + TallyUp, data=imputed)

summary(ZIP_Model)
summary(ZIP_Model$fitted.values)
AIC(ZIP_Model)
mse(ZIP_Model)

imputed$ZIPphat <- predict(ZIP_Model, newdata = imputed, type = "response")
ZIP_Model.test <- imputed$ZIPphat
ZIP_Model.R2 <- R2(ZIPModel.test, imputed$TARGET)
ZIP_Model.Adj_R2.test <- 1 - (1 - ZIP_Model.R2) * (nrow(imputed) - 1)/(nrow(imputed)-(length(ZIP_Model$coefficients)-1)-1)

##########################################################################################
##########################################################################################
## 6TH MODEL ... ZERO INFLATED NEGATIVE BINOMIAL REGRESSION (ZINB)########################
##########################################################################################

ZINB_Model<-zeroinfl(TARGET ~ STARS_IMP + STARS_IMP_FLAG + LabelAppeal_IMP + AcidIndex_IMP + 
                       ResidualSugar_IMP_REDFLAG + Density_IMP_REDFLAG + TallyUp, data=imputed, dist = "negbin", EM=TRUE)
summary(ZINB_Model)

AIC(ZINB_Model)
mse(ZINB_Model)

imputed$ZINBphat <- predict(ZINB_Model, newdata = imputed, type = "response")
ZINB_Model.test <- imputed$ZIPphat
ZINB_Model.R2 <- R2(ZIPModel.test, imputed$TARGET)
ZINB_Model.Adj_R2.test <- 1 - (1 - ZINB_Model.R2) * (nrow(imputed) - 1)/(nrow(imputed)-(length(ZINB_Model$coefficients)-1)-1)


#what type of dispersion does sample have?
mean(imputed$TARGET)
var(imputed$TARGET)


# Compare models
lm_fit.AIC <- AIC(lm_fit)
lm_fit.mse <- mse(lm_fit)
lm_fit_stepwise.AIC <- AIC(lm_fit_stepwise)
lm_fit_stepwise.mse <- mse(lm_fit_stepwise)
poisson_model.AIC <- AIC(NBR_Model)
poisson_model.mse <- mse(poisson_model)
NBR_Model.mse <- mse(NBR_Model)
NBR_Model.AIC <- AIC(NBR_Model)
NBR_Model.mse <- mse(NBR_Model)
ZIP_Model.AIC <- AIC(ZIP_Model)
ZIP_Model.mse <- mse(ZIP_Model)
ZINB_Model.AIC <- AIC(ZINB_Model)
ZINB_Model.mse <- mse(ZINB_Model)


# Table of Metrics
rnames <- c("lm_fit", "lm_fit_stepwise", "poisson_model", "NBR_Model", "ZIP_Model", "ZINB_Model")
mse_aic_table <- matrix(list(lm_fit.mse, lm_fit.Adj_R2.test,  lm_fit.AIC,
                             lm_fit_stepwise.mse, lm_fit_stepwise.Adj_R2.test,  lm_fit_stepwise.AIC, 
                             poisson_model.mse, poisson_model.Adj_R2.test,  poisson_model.AIC, 
                             NBR_Model.mse, NBR_Model.Adj_R2.test, NBR_Model.AIC, 
                             ZIP_Model.mse, ZIP_Model.Adj_R2.test, ZIP_Model.AIC,
                             ZINB_Model.mse, ZINB_Model.Adj_R2.test, ZINB_Model.AIC), ncol=3, byrow=TRUE)
colnames(mse_aic_table) <- c("MSE", "Adj_R2", "AIC")
mse_aic_table <- as.table(mse_aic_table)
mse_aic_table <- cbind(Row.Names = c(rnames), mse_aic_table)
mse_aic_table
