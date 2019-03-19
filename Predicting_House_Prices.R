
setwd("C:/Users/brench/Desktop/School/MSDS 410 Regression and Multivariate Analysis/Assignments/Assignments 1-5/Data")

mydata <- read.csv(file="ames_housing_data.csv",head=TRUE,sep=",")

# Import needed packages
require(corrplot)
require(ggplot2)
require(gridExtra)
require(car)
source("http://www.sthda.com/upload/rquery_cormat.r")
require(lattice)
library(tidyr)
library(moments)
library(plyr)


# Get overview of the data
str(mydata)
head(mydata)
names(mydata)

# Define these variables for later use;
mydata$QualityIndex <- mydata$OverallQual*mydata$OverallCond
mydata$TotalSF <- mydata$FirstFlrSF + mydata$SecondFlrSF+mydata$TotalBsmtSF
mydata$HouseAge <- mydata$YrSold - mydata$YearBuilt
mydata$GarageAge <- mydata$YrSold-mydata$GarageYrBlt

# List of numeric columns
nums <- unlist(lapply(mydata, is.numeric))  
numeric_cols <- colnames(mydata)
numeric_cols[nums]

# List of categorical columns
numeric_cols[!nums]

# Evalute data. Evaluate columns with NAs in numeric columns. Later we will deal with NAs
# for the categorical variables.
summary(mydata[,nums])

# For HouseAge we see -1. This is because the house sell year is 2007, but the built is 2008. We will change this is 0.
mydata$HouseAge[mydata$HouseAge==-1] <- 0
mydata$YearBuilt[mydata$HouseAge==-1] <- 2007


# After looking at categories and fixing wrongly input variables, lets split the data.
# Create train/test split;
set.seed(123)
mydata$u <- runif(n=dim(mydata)[1],min=0,max=1) #Creates random numbers that are uniformly distributed
trainset <- subset(mydata, u<0.70)
testset  <- subset(mydata, u>=0.70)

# Check your data split. The sum of the parts should equal the whole.
# Do your totals add up?
dim(mydata)[1]
dim(trainset)[1]
dim(testset)[1]
dim(trainset)[1]+dim(testset)[1]

obs_table <- matrix(c(dim(mydata)[1],
                      dim(trainset)[1],
                      dim(testset)[1],
                      dim(trainset)[1]+dim(testset)[1]),ncol=4,byrow=TRUE)
colnames(obs_table) <- c("Obs. of initial DS","Train","Test", "train+test")
obs_table <- as.table(obs_table)
obs_table

# Compute correlation matrix. Could instead compute heatma but due to the # of variables, its 
# clearer to use size of the symbol to display the strength of the dependence.
# rquery returns the following: 
# r : The table of correlation coefficients
# p : Table of p-values corresponding to the significance levels of the correlations
# sym : A representation of the correlation matrix in which coefficients are replaced by 
# symbols according to the strength of the dependence. For more description, see this article: 
# Visualize correlation matrix using symnum function
rquery.cormat(trainset[,nums])

# Correlation with response variable Sale Price
corr_table <- as.data.frame(cor(trainset[,nums],trainset$SalePrice, use = "pairwise.complete.obs"))
corr_table[order(-corr_table$V1), , drop=FALSE]
# Keep in mind these are the correlations of the entire data set (not just train) and before 
# any transformation or imputing. Howver, this provides us with a sense of what variables 
# are worth further investigation.


########################### EDA on Continous/Discrete Variables ############################
# Evaluate NAs
na_df <- data.frame(colSums(is.na(trainset)))
colnames(na_df)[1] <- "count_nas"
na_df$percent_of_observations <- round(na_df$count_nas/nrow(trainset),3)
na_df <- na_df[rowSums(na_df) > 0,]
na_df

# Total SF has one NA which means that trainset$TotalBsmt has an NA and should be a 0. 
# The same could be said of several variables but we aren't using all of these. So we 
# will only transform ones we might use: BmtFullBath, GarageArea, GarageCars 
trainset$TotalBsmtSF[is.na(trainset$TotalBsmtSF)] <- 0
trainset$BsmtFullBath[is.na(trainset$BsmtFullBath)] <- 0
trainset$GarageArea[is.na(trainset$GarageArea)] <- 0
trainset$GarageCars[is.na(trainset$GarageCars)] <- 0
trainset$MasVnrArea[is.na(trainset$MasVnrArea)] <- 0

#Since we defined TotalSF before imputing missing values, let's run it again.
trainset$TotalSF <- trainset$FirstFlrSF + trainset$SecondFlrSF+trainset$TotalBsmtSF

# We also see that PoolQC, MiscFeature, Alley, Fence and Fireplace QU have high counts of NAs and wont
# be effective predictors. Lot frontage has 17% NAs which is also high.

# Lot Frontage. We could impute the mean or check correlation with Lot Area which has no missing values.
cor(trainset$LotFrontage, trainset$LotArea, use = "pairwise.complete.obs")
# Most lots are rectangular so let's try the square root
cor(trainset$LotFrontage, sqrt(trainset$LotArea), use = "pairwise.complete.obs")
# Let's see what this transformation does to LotArea

par(mfrow=c(1,2))
hist(trainset$LotArea, col = "lightblue")
hist(sqrt(trainset$LotArea), col = "red")
par(mfrow=c(1,1))
# The transformation makes sense and we could impute the square root of LotArea for a given observation
# However, should be just be using LotArea instead?
cor(trainset$LotFrontage,trainset$SalePrice, use="pairwise.complete.obs")
cor(trainset$LotArea,trainset$SalePrice)
cor(sqrt(trainset$LotArea),trainset$SalePrice)
# The correlation is similar when we use the square root so we will go ahead with simply using
trainset$LotArea <- sqrt(trainset$LotArea)
testset$LotArea <- sqrt(testset$LotArea)

# Evaluate NAs for GarageAge
summary(trainset$GarageAge)
# -200 can't be correct and there are lots of NAs.
trainset$GarageAge[is.na(trainset$GarageAge)] <- 0
trainset[which(trainset$GarageAge <= -1),"GarageAge"] # this is the only one larger than 2010.
# The house built date was 2006 so lets change to 06
trainset$GarageAge[trainset$GarageAge == -200] <- (trainset$YrSold[trainset$GarageAge == -200] - trainset$YearBuilt[trainset$GarageAge == -200])

# TotalBsmtSF
trainset$TotalBsmtSF[is.na(trainset$TotalBsmtSF)] <- 0  
cor(trainset$GarageYrBlt,trainset$YearBuilt,use="pairwise.complete.obs")
trainset$GarageYrBlt[is.na(trainset$GarageYrBlt)] <- trainset$YearBuilt[is.na(trainset$GarageYrBlt)]
summary(trainset$GarageYrBlt)
cor(trainset$GarageYrBlt,trainset$SalePrice)

# Investigate Response Variable
hist(trainset$SalePrice, main = "Distribution of Sale Price", col="lightgreen")
hist(log(trainset$SalePrice), main = "Distribution of Sale Price", col="dodgerblue2")
# For our purposes, we will continue with non-log transformed SalePrice.

# We will use the following Continuous/discrete variables for our model: OverallQual, TotalSF, 
# GarageCars, YearBuilt (.558) or HouseAge,  FullBath, YearRemodel, GarageYrBlt, 
# MasVnrArea, TotalRmsAbvGrd, Fireplaces, Lot Area, WoodDeckS.

# Create pairplots of selected variables
#pairs(trainset[,c("OverallQual", "TotalSF", "GarageCars", "HouseAge", 
# "FullBath", "YearRemodel", "MasVnrArea",
# "TotRmsAbvGrd", "Fireplaces", "LotArea", "WoodDeckSF", "SalePrice")], pch = 21)

# Create distributions of each variable
par(mfrow=c(2,3))
hist(trainset$OverallQual)
hist(trainset$TotalSF)
hist(trainset$GarageCars)
hist(trainset$HouseAge)
hist(trainset$FullBath)
hist(trainset$YearRemodel)

par(mfrow=c(2,3))
hist(trainset$GarageYrBlt)
hist(trainset$MasVnrArea)
hist(trainset$TotRmsAbvGrd)
hist(trainset$Fireplaces)
hist(trainset$LotArea)
hist(trainset$WoodDeckSF)

# Its possible that some of these will need transformations. Let's start by viewing
# the residuals of Total SF by fitting a lm.

MLRresult <- lm(SalePrice ~ TotalSF + OverallQual, data=trainset)
anova(MLRresult)
summary(MLRresult)

par(mfrow=c(2,2))
plot(MLRresult)

pred <- as.data.frame(predict(MLRresult, trainset))
names(pred)
pred <- rename(pred, c("predict(MLRresult, trainset)" = "prd"))
trainset_df <- trainset
trainset_df$pred <- pred$prd
summary(pred)
trainset_df$res <- trainset_df$SalePrice - trainset_df$pred
trainset_df$absres <- abs(trainset_df$res)
ggplot(trainset_df, aes(x=Neighborhood , y=res)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of Residuals by Neighborhood ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(trainset_df, aes(x=Neighborhood , y=SalePrice)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of Residuals by Neighborhood ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

summary(trainset_df$absres)
MAE <- mean(trainset_df$absres)
MAE


########################### Evaluate Needed Transformations ############################
# We see some issues with the residuals. However, the assignment suggested
# not to make transformation to SalePrice. I investigating making transformations of 
# the predictors but this did not fix the residuals. An example is shown below - I'm 
# not positive if I'm applyying boxTidwell correctly since in this case I'm really trying
# to transform the predictor. In any case, I don't end up taking the transformations.

# Transformation to TotalSF
x <- trainset$SalePrice
y <- trainset$TotalSF # Normally this would be SalePrice but to find the transformation of
# the predictor this is how I had to code it... suggestions?
cor(y,x)
hist(y)
skewness(y)
kurtosis(y)

boxTidwell(y~x)
scores <- boxTidwell(y~x)
lambda <- scores$result[3]
cor(y^lambda,x)
hist(y^lambda)
skewness(y^lambda)
kurtosis(y^lambda) # Still higher than preferred

m <- lm(y ~ x)
summary(m)
# re-run with transformation
y_after <- y^lambda
mnew <- lm(y_after ~ x)
summary(mnew)

# QQ-plot of residuals
op <- par(pty = "s", mfrow = c(1, 2))
qqnorm(m$residuals); qqline(m$residuals)
qqnorm(mnew$residuals); qqline(mnew$residuals)
par(op)  # not really improved which we will see later.

# Apply transformation
trainset_df$TotalSFTransformed <- trainset$TotalSF^lambda

MLRresult <- lm(SalePrice ~ TotalSFTransformed + OverallQual, data=trainset)
anova(MLRresult)
summary(MLRresult)

par(mfrow=c(2,2))
plot(MLRresult)
# We aren't seeing improvements in the residuals so we don't keep transformation.

# Since we aren't making transformations, we can go ahead and use trainset moving forward

########################### EDA on Categorical Values ############################
# Let's evaluate categorical values.
summary(trainset[,!nums])
tapply(trainset$SalePrice, trainset$HouseStyle, mean)
tapply(trainset$SalePrice, trainset$SaleCondition, mean)

# Based on the variety of observations in different categories some don't look promising.
# We can also evaluate if there are large differences in mean SalePrice between Categories

summary(trainset$HouseStyle) 
ggplot(trainset, aes(x=HouseStyle , y=SalePrice)) + 
  geom_boxplot(fill="green") +
  labs(title="Distribution of Residuals by Condition ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

# Due to limited observations and closer means, combine 1.5fin and 1.5unf, 2.5 fin, 2.5unf
trainset$HouseStyle <- recode(trainset$HouseStyle, "c('1.5Fin','1.5Unf')='1.5'")
trainset$HouseStyle <- recode(trainset$HouseStyle, "c('2.5Fin','2.5Unf')='2.5'")
summary(trainset$HouseStyle)

tapply(trainset$SalePrice, trainset$LotShape, mean)
summary(trainset$LotShape) 
ggplot(trainset, aes(x=LotShape , y=SalePrice)) + 
  geom_boxplot(fill="red") +
  labs(title="Distribution of Residuals by Condition ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

# Combine IR1-IR3 and keep Reg.
trainset$LotShape <- recode(trainset$LotShape, "c('IR1','IR2', 'IR3')='IR'; else='Reg'")
summary(trainset$LotShape)

ggplot(trainset, aes(x=SaleCondition , y=SalePrice)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of Residuals by Condition ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))
summary(trainset$SaleCondition)

# The variable doesn't look helpful for making predictions. The bigger concern is if SaleCondition creates outliers.
# abnormal has 190 obs with 10 outliers, partial has 245 with 4. Will these be problematic?

ggplot(trainset, aes(x=Neighborhood , y=SalePrice)) + 
  geom_boxplot(fill="blue") +
  labs(title="Distribution of Residuals by Neighborhood ") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

# We have many neighborhoods with few observations. We need to either combine neighborhoods 
# with similar mean SalePrice or find a highly correlated variable that can be used 
# to group these. We will try price/sqft.

trainset$PriceSF <- trainset$SalePrice/trainset$TotalSF
testset$PriceSF <- testset$SalePrice/testset$TotalSF

train.clean <- ddply(trainset, .(Neighborhood), summarize,
                     MAE = mean(absres))
trainset2 <- ddply(trainset, .(Neighborhood), summarize,
                   MeanPrice = mean(SalePrice))
trainset3 <- ddply(trainset, .(Neighborhood), summarize,
                   TotalPrice = mean(SalePrice))
trainset4 <- ddply(trainset, .(Neighborhood), summarize,
                   TotalSqft = mean(TotalSF))
trainset34 <- cbind(trainset3, trainset4)
trainset34$AvgPr_swft <- trainset34$TotalPrice/trainset34$TotalSqft

trainsetall <- train.clean
trainsetall$MeanPrice <- trainset2$MeanPrice
trainsetall$AvgPr_Swft <- trainset34$AvgPr_swft

# The trainset added 4 columns to investigate how to group categories 

require(ggplot2)
ggplot(trainsetall, aes(x=AvgPr_Swft, y=MeanPrice)) +
  geom_point(color="dodgerblue2", shape=1, size=3) +
  ggtitle("Scatterplot") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

cor(trainsetall$AvgPr_Swft, trainsetall$MeanPrice, use = "pairwise.complete.obs")
# We can now see where we want to group our categories.

# The trainset added 4 columns to investigate how to group categories 
# For simplicity sake we return to training data

# Create Neighborhood Groups
trainset$NbhdGrp <-
  ifelse(trainset$PriceSF<=60, "grp1",
         ifelse(trainset$PriceSF<=66, "grp2",
                ifelse(trainset$PriceSF<=78, "grp3", "grp4")))

testset$NbhdGrp <-
  ifelse(testset$PriceSF<=60, "grp1",
         ifelse(testset$PriceSF<=66, "grp2",
                ifelse(testset$PriceSF<=78, "grp3", "grp4")))


#Let's take a look at the errors by SaleCondition
summary(mydata$SaleCondition)
summary(trainset$absres)
MAE <- aggregate(trainset$absres, list(trainset$SaleCondition), mean)
MAE
# We can see that on partials we have larger residuals. Let's keep this in mind in case we should leave out partials.


########################### Define Dummy Varaibles ############################
trainset$NbhdGrp1 <-
  ifelse(trainset$NbhdGrp == "grp1", 1, 0)
trainset$NbhdGrp2 <-
  ifelse(trainset$NbhdGrp == "grp2", 1, 0)
trainset$NbhdGrp3 <-
  ifelse(trainset$NbhdGrp == "grp3", 1, 0)

testset$NbhdGrp1 <-
  ifelse(testset$NbhdGrp == "grp1", 1, 0)
testset$NbhdGrp2 <-
  ifelse(testset$NbhdGrp == "grp2", 1, 0)
testset$NbhdGrp3 <-
  ifelse(testset$NbhdGrp == "grp3", 1, 0)

summary(trainset$HouseStyle) 
# Due to limited observations and closer means, combine 1.5Fin and 1.5Unf, 2.5 fin, 2.5unf
trainset$HouseStyle <- recode(trainset$HouseStyle, "c('1.5Fin','1.5Unf')='1.5'")
trainset$HouseStyle <- recode(trainset$HouseStyle, "c('2.5Fin','2.5Unf')='2.5'")
summary(trainset$HouseStyle)

tapply(trainset$SalePrice, trainset$LotShape, mean)
summary(trainset$LotShape) 
# Combine IR1-IR3 and keep Reg.
trainset$LotShape <- recode(trainset$LotShape, "c('IR1','IR2', 'IR3')='IR'; else='Reg'")
summary(trainset$LotShape)

trainset$HouseStyle1 <-
  ifelse(trainset$HouseStyle == "1.5", 1, 0)
trainset$HouseStyle2 <-
  ifelse(trainset$HouseStyle == "SFoyer", 1, 0)
trainset$HouseStyle3 <-
  ifelse(trainset$HouseStyle == "1Story", 1, 0)
trainset$HouseStyle4 <-
  ifelse(trainset$HouseStyle == "2.5", 1, 0)

testset$HouseStyle1 <-
  ifelse(testset$HouseStyle == "1.5", 1, 0)
testset$HouseStyle2 <-
  ifelse(testset$HouseStyle == "SFoyer", 1, 0)
testset$HouseStyle3 <-
  ifelse(testset$HouseStyle == "1Story", 1, 0)
testset$HouseStyle4 <-
  ifelse(testset$HouseStyle == "2.5", 1, 0)

# LotShape dummy encoding
trainset$LotShape1 <-
  ifelse(trainset$LotShape == "IR", 1, 0)

testset$LotShape1 <-
  ifelse(testset$LotShape == "IR", 1, 0)

colnames(trainset)
# Let's create dataframe with predictor variables
drop.list <- c('SID','PID','SubClass', 'Zoning', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour',
               'Utilities','LotConfig','LandSlope', 'Neighborhood', 'BedroomAbvGr', 
               'GarageType', 'Condition1','Condition2', 'BldgType', 'HouseStyle','RoofStyle', 'RoofMat', 'GarageCond', 'PavedDrive', 'OpenPorchSF', 'res',
               'Exterior1', 'Exterior2', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
               'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FirstFlrSF', 'SecondFlrSF', 'LowQualFinSF','YrSold','MoSold','SaleCondition',
               'u','train','I2010','BsmtFullBath','BsmtHalfBath','HalfBath', 'KitchenAbvGr', 'KitchenQual', 'Functional',
               'FireplaceQu', 'Garagetype', 'GarageYrBlt', 'GarageQual', 'GarageFinish', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea',
               'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'SaleType', 'QualityIndex','GarageArea', 'YearBuilt', 'GarageAge','PriceSF','NbhdGrp',
               'FireplaceInd1','FireplaceInd2','OverallCond','PoolArea','GrLivArea');

train.clean <-trainset[,!(names(trainset) %in% drop.list)];
colnames(train.clean)

# train.clean <- trainset[c("SalePrice", "OverallQual", "TotalSF", "GarageCars", "HouseAge", 
#                           "FullBath", "YearRemodel", "MasVnrArea",
#                           "TotRmsAbvGrd", "Fireplaces", "LotArea", "WoodDeckSF", 
#                           "HouseStyle1", "HouseStyle2", "HouseStyle3", "HouseStyle4", "LotShape1",
#                           "NbhdGrp1", "NbhdGrp2", "NbhdGrp3")]

variable_table <- matrix(c(7, 5, 3),ncol=3,byrow=TRUE)
colnames(variable_table) <- c("Continuous", "Discrete", "Categorical")
variable_table <- as.table(variable_table)
variable_table

test.clean <- testset[c("SalePrice", "OverallQual", "TotalSF", "GarageCars", "HouseAge", 
                        "FullBath", "YearRemodel", "MasVnrArea",
                        "TotRmsAbvGrd", "Fireplaces", "LotArea", "WoodDeckSF", 
                        "HouseStyle1", "HouseStyle2", "HouseStyle3", "HouseStyle4", "LotShape1",
                        "NbhdGrp1", "NbhdGrp2", "NbhdGrp3")]

########################### Build Model ##################################

train.clean <- na.omit(train.clean) # There aren't any missing values, but just to be thorough
test.clean <- na.omit(test.clean)

junk.lm <- lm(SalePrice ~ QualityIndex + ExterQual + Functional + Zoning + GarageType, data=trainset)
summary(junk.lm)

#Define the upper model as the full model
upper.lm <- lm(SalePrice ~ ., data=train.clean);
summary(upper.lm)

# Define the lower model as the intercept model
lower.lm <- lm(SalePrice ~ -1, data = train.clean);
summary(lower.lm)
# Need a SLR to initialize stepwise selection
sqft.lm <- lm(SalePrice ~ TotalSF, data = train.clean);
summary(sqft.lm)

library(MASS)
# StepAIC() is part of the MASS library
# Call stepAIC() for variable selection
forward.lm <- stepAIC(object=lower.lm, scope=list(upper=upper.lm, lower = lower.lm),
                      direction=c('forward'));
summary(forward.lm)

backward.lm <- stepAIC(object=upper.lm, direction=c('backward'));
summary(backward.lm)

stepwise.lm <- stepAIC(object=sqft.lm, scope=list(upper=formula(upper.lm),lower= lower.lm),
                       direction=c('both'));
summary(stepwise.lm)

sort(vif(forward.lm),decreasing=TRUE) # Without intercept the calculation then we will have an issue.
sort(vif(backward.lm),decreasing=TRUE) 
sort(vif(stepwise.lm),decreasing=TRUE)
# None of the VIF values are too concerning.


########################### Removing Outliers ############################

# Plot residuals
par(mfrow=c(2,2))
plot(stepwise.lm)

train.clean1 <- train.clean
pred <- predict(stepwise.lm, train.clean1, interval = "prediction" )
train.clean1$res <- train.clean1$SalePrice-pred
summary(abs(train.clean1$res))
MAE <- mean(abs(train.clean1$res))
MAE

library(car)
vif(stepwise.lm)
par(mfrow=c(1,1))
#influencePlot(stepwise.lm, id.method = "identify", main="Influence Plot",
#              sub="Circle size is proportional to Cook's Distance")


summary(inflm.stepwise_lm <- influence.measures(stepwise.lm))

# install.packages("olsrr")
dffit_score <- dffits(stepwise.lm)
summary(stepwise.lm)
train.clean1 <- cbind(train.clean1, dffit_score)
str(train.clean1)

summary(train.clean1)
# Largeset absolute dffit_score
head(sort(abs(train.clean1$dffit_score)),5)

ols_plot_dffits(MLR)
train.clean1$absdf <- abs(train.clean1$dffit_score)
head(train.clean1)
summary(train.clean1)
p <- 16
n <- length(train.clean1$absdf)
dffits <- 2*(sqrt((p+1)/(n-p-1)))
trainset_inf <- train.clean1[which(train.clean1$absdf < dffits),]

# Instead of removing we can take a look at them and determine whether to remove but I'm leaving it as is.
# After removing influenctial points let's take a look at outliers.
#pairs(trainset_inf[,c("OverallQual", "TotalSF", "GarageCars", "HouseAge", 
#                      "FullBath", "YearRemodel","MasVnrArea",
#                      "TotRmsAbvGrd", "Fireplaces", "LotArea", "WoodDeckSF", "SalePrice")], pch = 21)
# We can still see some outliers so let's just make sure there are no isses.
hist(trainset_inf$SalePrice)
ggplot(trainset_inf, aes(x=OverallQual, y=SalePrice)) + 
  geom_point(color="blue", shape=1) +
  ggtitle("Scatter Plot of Total Floor SF vs QualityIndex") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))
# This looks really good.
hist(trainset_inf$WoodDeckSF)
summary(trainset_inf$WoodDeckSF)
ggplot(trainset_inf, aes(x=WoodDeckSF, y=SalePrice)) + 
  geom_point(color="blue", shape=1) +
  ggtitle("Scatter Plot of Total Floor SF vs QualityIndex") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))
# We see an outlying point but looking at the row we don't really see too much of a problem.

hist(trainset_inf$LotArea)
summary(trainset_inf$LotArea)
ggplot(trainset_inf, aes(x=LotArea, y=SalePrice)) + 
  geom_point(color="blue", shape=1) +
  ggtitle("Scatter Plot of Total Floor SF vs QualityIndex") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))
# We see two outlying points. They didn't come up as influential points, let's take a look at their dffits values
trainset_inf[which(trainset_inf$LotArea > 300),] # Both are fairly low and so they don't seem to be making a huge impact.

hist(trainset_inf$MasVnrArea)
summary(trainset_inf$MasVnrArea)
ggplot(trainset_inf, aes(x=MasVnrArea, y=SalePrice)) + 
  geom_point(color="blue", shape=1) +
  ggtitle("Scatter Plot of Total Floor SF vs QualityIndex") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))
trainset_inf[which(trainset_inf$MasVnrArea > 1000),] # 2259 Is fairly close to the dffits value of .18, but I'll leave it

# redefine model in order to make predictions
train.clean1 <- trainset_inf[c("SalePrice", "OverallQual", "TotalSF", "GarageCars", "HouseAge", 
                               "FullBath", "YearRemodel", "MasVnrArea",
                               "TotRmsAbvGrd", "Fireplaces", "LotArea", "WoodDeckSF", 
                               "HouseStyle1", "HouseStyle2", "HouseStyle3", "HouseStyle4", "LotShape1",
                               "NbhdGrp1", "NbhdGrp2", "NbhdGrp3")]


########################### Rebuilding Model ############################
#Define the upper model as the full model
upper.lm <- lm(SalePrice ~ ., data=train.clean1);
summary(upper.lm)

# Define the lower model as the intercept model
lower.lm <- lm(SalePrice ~ -1, data = train.clean1);
summary(lower.lm)
# Need a SLR to initialize stepwise selection
sqft.lm <- lm(SalePrice ~ TotalSF, data = train.clean1);
summary(sqft.lm)

# StepAIC() is part of the MASS library
# Call stepAIC() for variable selection
forward.lm <- stepAIC(object=lower.lm, scope=list(upper=upper.lm, lower = lower.lm),
                      direction=c('forward'));
summary(forward.lm)

backward.lm <- stepAIC(object=upper.lm, direction=c('backward'));
summary(backward.lm)

stepwise.lm <- stepAIC(object=sqft.lm, scope=list(upper=formula(upper.lm),lower= lower.lm),
                       direction=c('both'));
summary(stepwise.lm)

sort(vif(forward.lm),decreasing=TRUE) # Without intercept the calculation then we will have an issue.
sort(vif(backward.lm),decreasing=TRUE) 
sort(vif(stepwise.lm),decreasing=TRUE)
# None of the VIF values are too concerning.


R_squared_forward <- 1 - anova(forward.lm)["Residuals", "Sum Sq"]/ sum(anova(forward.lm)["Sum Sq"])
R_squared_backward <- 1 - Anova(backward.lm)["Residuals", "Sum Sq"]/ sum(Anova(backward.lm)["Sum Sq"])
R_squared_stepwise <- 1 - Anova(stepwise.lm)["Residuals", "Sum Sq"]/ sum(Anova(stepwise.lm)["Sum Sq"])
R_squared_forward 
R_squared_backward 
R_squared_stepwise 

AIC_forward <- 2*13*(1972+13)*(log(deviance(forward.lm))/(1972+13))
AIC_backward <- 2*16*(1969+16)*(log(deviance(backward.lm))/(1972+16))
AIC_stepwise <- 2*16*(1969+16)*(log(deviance(stepwise.lm))/(1972+16))
AIC_forward
AIC_backward
AIC_stepwise

BIC_forward <- 1972*(log(deviance(forward.lm))/(1972)) + 13*log(1972)
BIC_backward <- 1969*(log(deviance(backward.lm))/(1969)) + 16*log(1969)
BIC_stepwise <- 1969*(log(deviance(stepwise.lm))/(1969)) + 16*log(1969)
BIC_forward
BIC_backward
BIC_stepwise

# Calculate RMS and MAE
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

actual <- train.clean1$SalePrice
predicted <- predict(forward.lm)
error_forward <- actual - predicted
rmse(error_forward)
mae(error_forward)

actual <- train.clean1$SalePrice
predicted <- predict(backward.lm)
error_backward <- actual - predicted
rmse(error_backward)
mae(error_backward)

actual <- train.clean1$SalePrice
predicted <- predict(stepwise.lm)
error_stepwise <- actual - predicted
rmse(error_stepwise)
mae(error_stepwise)


########################### Test Model ############################
forward.test <- predict(forward.lm,newdata=test.clean);
backward.test <- predict(backward.lm,newdata=test.clean);
stepwise.test <- predict(stepwise.lm,newdata=test.clean);

# Training Data
# Abs Pct Error
forward.pct <- abs(forward.lm$residuals)/train.clean1$SalePrice; #Abs error/actual value so % error
MAPE <- mean(forward.pct)  # Mean of % abs error.
MAPE # 10% +- error of salePrice

backward.pct <- abs(backward.lm$residuals)/train.clean1$SalePrice;
MAPE <- mean(forward.pct)
MAPE

stepwise.pct <- abs(stepwise.lm$residuals)/train.clean1$SalePrice;
MAPE <- mean(forward.pct)
MAPE

# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice - forward.test)/test.clean$SalePrice;  # The predicted Sale Price using the forward test
backward.testPCT <- abs(test.clean$SalePrice - backward.test)/test.clean$SalePrice;
stepwise.testPCT <- abs(test.clean$SalePrice - stepwise.test)/test.clean$SalePrice;

MAPE <- mean(forward.testPCT)
MAPE
backward.testPCT <- abs(test.clean$SalePrice - backward.test)/test.clean$SalePrice;
MAPE <- mean(forward.testPCT)
MAPE
stepwise.test <- abs(test.clean$SalePrice - stepwise.test)/test.clean$SalePrice;
MAPE <- mean(forward.testPCT)
MAPE


# Calculate RMS and MAE
rmse <- function(error)
{
  sqrt(mean(error^2))
}

actual <- test.clean$SalePrice
predicted <- forward.test
error_forward <- actual - predicted
rmse(error_forward)
mae(error_forward)

actual <- test.clean$SalePrice
predicted <- backward.test
error_backward <- actual - predicted
rmse(error_backward)
mae(error_backward)

actual <- test.clean$SalePrice
predicted <- stepwise.test
error_stepwise <- actual - predicted
rmse(error_stepwise)
mae(error_stepwise)


# Assign Prediction Grades training data;
forward.PredictionGrade <- ifelse(forward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(forward.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(forward.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)

backward.PredictionGrade <- ifelse(backward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(backward.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(backward.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)

stepwise.PredictionGrade <- ifelse(stepwise.testPCT<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(stepwise.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(stepwise.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)

forward.trainTable <- table(forward.PredictionGrade)
forward.trainTable/sum(forward.trainTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+
backward.trainTable <- table(backward.PredictionGrade)
backward.trainTable/sum(backward.trainTable) 
stepwise.trainTable <- table(stepwise.PredictionGrade)
stepwise.trainTable/sum(stepwise.trainTable) 

# Assign Prediction Grades training data;
forward.PredictionGrade <- ifelse(forward.pct<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(forward.pct<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(forward.pct<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)
forward.trainTable <- table(forward.PredictionGrade)
forward.trainTable/sum(forward.trainTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+

# Assign Prediction Grades test data;
forward.testPredictionGrade <- ifelse(forward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                      ifelse(forward.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                             ifelse(forward.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                    'Grade 4: (0.25+]')
                                      )   
)
forward.testTable <- table(forward.testPredictionGrade)
forward.testTable/sum(forward.testTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+


# Assign Prediction Grades training data;
backward.PredictionGrade <- ifelse(backward.pct<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(backward.pct<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(backward.pct<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)
backward.trainTable <- table(backward.PredictionGrade)
backward.trainTable/sum(backward.trainTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+

# Assign Prediction Grades test data;
backward.testPredictionGrade <- ifelse(backward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                      ifelse(backward.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                             ifelse(backward.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                    'Grade 4: (0.25+]')
                                      )   
)
backward.testTable <- table(backward.testPredictionGrade)
backward.testTable/sum(backward.testTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+

# Assign Prediction Grades training data;
stepwise.PredictionGrade <- ifelse(stepwise.pct<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(stepwise.pct<=0.15, 'Grade2: (0.10,0.15]',
                                         ifelse(stepwise.pct<=0.25, 'Grade3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )   
)
stepwise.trainTable <- table(stepwise.PredictionGrade)
stepwise.trainTable/sum(stepwise.trainTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+

# Assign Prediction Grades test data;
stepwise.testPredictionGrade <- ifelse(stepwise.testPCT<=0.10,'Grade 1: [0.0.10]',
                                      ifelse(stepwise.testPCT<=0.15, 'Grade2: (0.10,0.15]',
                                             ifelse(stepwise.testPCT<=0.25, 'Grade3: (0.15,0.25]',
                                                    'Grade 4: (0.25+]')
                                      )   
)
stepwise.testTable <- table(stepwise.testPredictionGrade)
stepwise.testTable/sum(stepwise.testTable)  # How many houses can we predict within 10%, 15%, 25%, 25%+




fw_test <- lm(formula = SalePrice ~ TotalSF + NbhdGrp1 + NbhdGrp2 + NbhdGrp3 + 
                OverallQual + MasVnrArea + LotArea + HouseStyle4 + WoodDeckSF + 
                FullBath + GarageCars + Fireplaces + HouseStyle3 - 1, data = test.clean)

anova(fw_test)
summary(fw_test)


back_test <- lm(formula = SalePrice ~ TotalSF + NbhdGrp1 + NbhdGrp2 + NbhdGrp3 + 
                  OverallQual + MasVnrArea + LotArea + HouseStyle4 + WoodDeckSF + 
                  FullBath + GarageCars + Fireplaces + HouseStyle3 - 1, data = test.clean)

# Same as above.
summary(back_test)
stepwise_test <- lm(formula = SalePrice ~ TotalSF + NbhdGrp1 + NbhdGrp2 + NbhdGrp3 + 
                      OverallQual + MasVnrArea + HouseStyle4 + LotArea + YearRemodel + 
                      FullBath + GarageCars + Fireplaces + WoodDeckSF + HouseStyle2 + 
                      HouseStyle3, data = test.clean)

anova(stepwise_test)
summary(stepwise_test)


RSS <- c(crossprod(fw_test$residuals))
MSE <- RSS / length(fw_test$residuals)
RMSE <- sqrt(MSE)
RMSE



