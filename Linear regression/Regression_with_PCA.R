
setwd("C:/Users/brench/Desktop/School/MSDS 410 Regression and Multivariate Analysis/Assignments/Assignments 6-8")


my.data <- read.csv(file="stock_portfolio.csv",head=TRUE,sep=",")

str(my.data)
head(my.data)
names(my.data)


# Note Date is a string of dd-Mon-yy in R this is '%d-%B-%y';
my.data$RDate <- as.Date(my.data$Date,'%d-%B-%y');
sorted.df <- my.data[order(my.data$RDate),];
head(sorted.df)

# In stock market data, the day to day prices will be correlated. So we will need to remove
# this auto correlations. The regression assumes the rows are independent. 
# We do this by taking the log of today - log of yestereday.
AA <- log(sorted.df$AA[-1]/sorted.df$AA[-dim(sorted.df)[1]]);
head(AA)
# Manually check the first entry: log(9.45/9.23)
# Type cast the array as a data frame;
returns.df <- as.data.frame(AA);
returns.df <- as.data.frame(AA);
str(returns.df)
returns.df$BAC <- log(sorted.df$BAC[-1]/sorted.df$BAC[-dim(sorted.df)[1]]);
returns.df$BHI <- log(sorted.df$BHI[-1]/sorted.df$BHI[-dim(sorted.df)[1]]);
returns.df$CVX <- log(sorted.df$CVX[-1]/sorted.df$CVX[-dim(sorted.df)[1]]);
returns.df$DD  <- log(sorted.df$DD[-1]/sorted.df$DD[-dim(sorted.df)[1]]);
returns.df$DOW <- log(sorted.df$DOW[-1]/sorted.df$DOW[-dim(sorted.df)[1]]);
returns.df$DPS <- log(sorted.df$DPS[-1]/sorted.df$DPS[-dim(sorted.df)[1]]);
returns.df$GS  <- log(sorted.df$GS[-1]/sorted.df$GS[-dim(sorted.df)[1]]);
returns.df$HAL <- log(sorted.df$HAL[-1]/sorted.df$HAL[-dim(sorted.df)[1]]);
returns.df$HES <- log(sorted.df$HES[-1]/sorted.df$HES[-dim(sorted.df)[1]]);
returns.df$HON <- log(sorted.df$HON[-1]/sorted.df$HON[-dim(sorted.df)[1]]);
returns.df$HUN <- log(sorted.df$HUN[-1]/sorted.df$HUN[-dim(sorted.df)[1]]);
returns.df$JPM <- log(sorted.df$JPM[-1]/sorted.df$JPM[-dim(sorted.df)[1]]);
returns.df$KO  <- log(sorted.df$KO[-1]/sorted.df$KO[-dim(sorted.df)[1]]);
returns.df$MMM <- log(sorted.df$MMM[-1]/sorted.df$MMM[-dim(sorted.df)[1]]);
returns.df$MPC <- log(sorted.df$MPC[-1]/sorted.df$MPC[-dim(sorted.df)[1]]);
returns.df$PEP <- log(sorted.df$PEP[-1]/sorted.df$PEP[-dim(sorted.df)[1]]);
returns.df$SLB <- log(sorted.df$SLB[-1]/sorted.df$SLB[-dim(sorted.df)[1]]);
returns.df$WFC <- log(sorted.df$WFC[-1]/sorted.df$WFC[-dim(sorted.df)[1]]);
returns.df$XOM <- log(sorted.df$XOM[-1]/sorted.df$XOM[-dim(sorted.df)[1]]);
returns.df$VV  <- log(sorted.df$VV[-1]/sorted.df$VV[-dim(sorted.df)[1]]);
str(returns.df) # Response variable
# Compute correlation matrix for returns;
returns.cor <- cor(returns.df)
returns.cor[,c('VV')]

# Barplot the last column to visualize magnitude of correlations;
barplot(returns.cor[1:20,c('VV')],las=2,ylim=c(0,1.0))
title('Correlations with VV')

# Make correlation plot for returns;
# If you need to install corrplot package;  Note how many dependencies this package has;
#install.packages('corrplot', dependencies=TRUE)

require(corrplot)
corrplot(returns.cor)

# load car package
require(car)

# Fit some model
model.1 <- lm(VV ~ MMM + HON +WFC + CVX + XOM, data=returns.df)
summary(model.1)
vif(model.1)


# Fit the full model
model.2 <- lm(VV ~ BAC+GS+JPM+WFC+BHI+CVX+DD+DOW+DPS+HAL+HES+HON+HUN+KO+MMM+MPC+PEP+SLB+XOM,data=returns.df)
summary(model.2)
vif(model.2)

# Fit an moderately sized model. Top 8 correlated variables
model.3 <- lm(VV ~ MMM+HON+WFC+CVX+XOM+DD+SLB+GS,data=returns.df)
summary(model.3)
vif(model.3)
anova(model.1)

AIC(model.1,model.2,model.3)
BIC(model.1,model.2,model.3)

#HAL took on a negative value but was postiively correlated with VV, 
# let's plot HAL and VV
require(ggplot2)
ggplot(returns.df, aes(x=HAL, y=VV)) + 
  geom_point(color="blue", shape=1) +
  ggtitle("Scatter Plot of HAL and VV") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

cor(returns.df$HAL,returns.df$VV)

returns.pca <- princomp(x=returns.df[,-21],cor=TRUE) # Extracts principal components
summary(returns.pca) # Provide overview of eigen values and cumulative variance exp
# -21 is due to leaving out VV: response variable.
# See the output components returned by princomp();
names(returns.pca)

pc.1 <- returns.pca$loadings[,1];
pc.2 <- returns.pca$loadings[,2];
names(pc.1)

dev.off()
plot(-10,10,type='p',xlim=c(0.27,-0.12),ylim=c(-0.27,0.6),xlab='PC 1',ylab='PC 2')
text(pc.1,pc.2,labels=names(pc.1), cex=0.75, pos =3) #Plots principal components eigen vector values.
points(pc.1,pc.2, pch = 18, cex=0.75, col=1:length(names(pc.1)))
legend(-.10,.62, cex=.75, legend = unique(names(pc.1)),col=1:length(names(pc.1)), pch=18)


# Plot the default scree plot;
plot(returns.pca, col = 'dodgerblue2', main = "Variance by Components") # How much variance is explained by each principal component.

# Make Scree Plot
scree.values <- (returns.pca$sdev^2)/sum(returns.pca$sdev^2);

plot(scree.values,xlab='Number of Components',ylab='',type='l',lwd=2)
points(scree.values,lwd=3,cex=1.5)
abline(h=0.025,lwd=1.5,col='red')
abline(v=8,lwd=1.5,col='red')
text(12,.05,'We start seeing very little difference',col='red')
title('Scree Plot')
# Another way of displaying the same results shown above. Looking for elbow.

# Make Proportion of Variance Explained
variance.values <- cumsum(returns.pca$sdev^2)/sum(returns.pca$sdev^2);

plot(variance.values,xlab='Number of Components',ylab='',type='l',lwd=2)
points(variance.values,lwd=2,cex=1.5)
abline(h=0.8,lwd=1.5,col='red')
abline(v=8,lwd=1.5,col='red')
text(13,0.5,'Keep 8 Principal Components',col='red')
title('Total Variance Explained Plot')

# Create the data frame of PCA predictor variables;
return.scores <- as.data.frame(returns.pca$scores);
return.scores$VV <- returns.df$VV;
return.scores$u <- runif(n=dim(return.scores)[1],min=0,max=1); # This is a uniform dist. of random scores 
# In order to split data.
head(return.scores)

# Split the data set into train and test data sets;
train.scores <- subset(return.scores,u<0.70);
test.scores <- subset(return.scores,u>=0.70);
dim(train.scores)
dim(test.scores)
dim(train.scores)+dim(test.scores)
dim(return.scores)

# Fit a linear regression model using the first 8 principal components;
pca1.lm <- lm(VV ~ Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6+Comp.7+Comp.8, data=train.scores);
summary(pca1.lm)
AIC(pca1.lm) # With fewer observations this cannot be compared to the AIC for the other models.
BIC(pca1.lm)


# Compute the Mean Absolute Error on the training sample;
pca1.mae.train <- mean(abs(train.scores$VV-pca1.lm$fitted.values));
vif(pca1.lm)

# Score the model out-of-sample and compute MAE;
pca1.test <- predict(pca1.lm,newdata=test.scores);
pca1.mae.test <- mean(abs(test.scores$VV-pca1.test));

# Let's compare the PCA regression model with a 'raw' regression model;
# Create a train/test split of the returns data set to match the scores data set;
returns.df$u <- return.scores$u;
train.returns <- subset(returns.df,u<0.70);
test.returns <- subset(returns.df,u>=0.70);
dim(train.returns)
dim(test.returns)
dim(train.returns)+dim(test.returns)
dim(returns.df)


# Fit model.1 on train data set and 'test' on test data;
model.1 <- lm(VV ~ GS+DD+DOW+HON+HUN+JPM+KO+MMM+XOM, data=train.returns)
model1.mae.train <- mean(abs(train.returns$VV-model.1$fitted.values));
model1.test <- predict(model.1,newdata=test.returns);
model1.mae.test <- mean(abs(test.returns$VV-model1.test));
summary(model.1)

# Fit model.2 on train data set and 'test' on test data;
model.2 <- lm(VV ~ BAC+GS+JPM+WFC+BHI+CVX+DD+DOW+DPS+HAL+HES+HON+HUN+KO+MMM+MPC+PEP+SLB+XOM, data=train.returns)
model2.mae.train <- mean(abs(train.returns$VV-model.2$fitted.values));
model2.test <- predict(model.2,newdata=test.returns);
model2.mae.test <- mean(abs(test.returns$VV-model2.test));
summary(model.2)

# remove u

train.scores <- train.scores[c(-22)]

# Fit full.lm on PCA scores of train data
full.lm <- lm(VV ~ ., data=train.scores);
summary(full.lm)

# Table of MAEs
rnames <- c("PCA1", "Model1", "Model2")
mae_table <- matrix(list(pca1.mae.train, pca1.mae.test, model1.mae.train, model1.mae.test, model2.mae.train,
                      model2.mae.test), ncol=2,byrow=TRUE)
colnames(mae_table) <- c("Train","Test")
mae_table <- as.table(mae_table)
mae_table <- cbind(Row.Names = c(rnames), mae_table)
mae_table

library(MASS)
backward.lm <- stepAIC(full.lm,direction=c('backward'))
summary(backward.lm)
backward.mae.train <- mean(abs(train.scores$VV-backward.lm$fitted.values));
vif(backward.lm)

backward.test <- predict(backward.lm,newdata=test.scores);
backward.mae.test <- mean(abs(test.scores$VV-backward.test))

mae_table <- rbind(mae_table, list("BackModel",backward.mae.train, backward.mae.test))

#Define the upper model as the full model
upper.lm <- lm(VV ~ ., data=train.scores);
summary(upper.lm)

# Define the lower model as the intercept model
lower.lm <- lm(VV ~ -1, data = train.scores);
summary(lower.lm)
# Need a SLR to initialize stepwise selection
sqft.lm <- lm(VV ~ Comp.1, data = train.scores);
summary(sqft.lm)

stepwise.lm <- stepAIC(object=sqft.lm, scope=list(upper=formula(upper.lm),lower= lower.lm),
                       direction=c('both'));
summary(stepwise.lm)
stepwise.lm.mae.train <- mean(abs(train.returns$VV-stepwise.lm$fitted.values));
stepwise.lm.test <- predict(model.2,newdata=test.returns);
stepwise.lm.mae.test <- mean(abs(test.returns$VV-stepwise.lm.test))

forward.lm <- stepAIC(object=lower.lm, scope=list(upper=upper.lm, lower = lower.lm),
                      direction=c('forward'));
summary(forward.lm)
forward.lm.mae.train <- mean(abs(train.returns$VV-forward.lm$fitted.values));
forward.lm.test <- predict(model.2,newdata=test.returns);
forward.lm.mae.test <- mean(abs(test.returns$VV-forward.lm.test))

similar.lm <- lm(VV ~ Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.8 + Comp.9 + Comp.10 + Comp.11, data = train.scores)
summary(forward.lm)
similar.lm.mae.train <- mean(abs(train.returns$VV-similar.lm$fitted.values));
similar.lm.test <- predict(model.2,newdata=test.returns);
similar.lm.mae.test <- mean(abs(test.returns$VV-similar.lm.test))

# AICs and BICs of all models
AIC(model.1, pca1.lm, similar.lm, backward.lm, full.lm)
BIC(model.1, pca1.lm, similar.lm, backward.lm, full.lm)

mae_table <- rbind(mae_table, list("Stepwise",stepwise.mae.train, stepwise.mae.test))
mae_table <- rbind(mae_table, list("forward",forward.lm.mae.train, forward.lm.mae.test))
mae_table <- rbind(mae_table, list("similar",similar.lm.mae.train, similar.lm.mae.test))
mae_table

