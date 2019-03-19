######## Download appropriate packages
library(rJava)
library(readr)
library(pbkrtest)
library(car)
library(leaps)
library(MASS)
library(xlsxjars)
library(xlsx)
library(moments) # Evaluate skeweness, kurtosis
library(tidyr)
library(DAAG) # k-fold validation
require(ggplot2)

setwd("D:/MSDS/411 GLM/MoneyBall Assignment")
moneyball <- read.csv("moneyball.csv",header=T)
moneyball_test <- read.csv(file="moneyball_test.csv",head=TRUE,sep=",")

############################# Part 1: Data Exploration ##################################################################
str(moneyball)
summary(moneyball)
nrows <- sapply( moneyball, function(f) 
sapply(moneyball$col, function(x) sum(length(which(is.na(x))))) 

# Wins - Use lower bound for lower outliers, upper bound for higher outliers.
par(mfrow=c(1,2))
hist(moneyball$TARGET_WINS, col = "#A71930", xlab = "TARGET_WINS", main = "Histogram of Wins")
boxplot(moneyball$TARGET_WINS, col = "#A71930", main = "Boxplot of Wins")
par(mfrow = c(1,1))

################# Batting ####################
# Hits and Doubles
par(mfrow=c(2,2))
hist(moneyball$TEAM_BATTING_H, col = "#A71930", xlab = "Team_Batting_H", main = "Histogram of Hits")
hist(moneyball$TEAM_BATTING_2B, col = "#09ADAD", xlab = "Doubles", main = "Histogram of Doubles")
boxplot(moneyball$TEAM_BATTING_H, col = "#A71930", main = "Boxplot of Hits")
boxplot(moneyball$TEAM_BATTING_2B, col = "#09ADAD", main = "Boxplot of Doubles")

# Triples and Home Runs
par(mfrow=c(2,2))
hist(moneyball$TEAM_BATTING_3B, col = "#A71930", xlab = "Triples", main = "Histogram of Triples", breaks = 20)
hist(moneyball$TEAM_BATTING_HR, col = "#DBCEAC", xlab = "Home Runs", main = "Histogram of Home Runs")
boxplot(moneyball$TEAM_BATTING_3B, col = "#A71930", main = "Boxplot of Triples")
boxplot(moneyball$TEAM_BATTING_HR, col = "#DBCEAC", main = "Boxplot of Home Runs")
par(mfrow=c(1,1))

# Triples, right skewed. Try transformation
# HR non-normal. 

# Walks, Strikeouts, HBP
par(mfrow=c(2,3))
hist(moneyball$TEAM_BATTING_BB, col = "#A71930", xlab = "Walks", main = "Histogram of Walks")
hist(moneyball$TEAM_BATTING_SO, col = "#09ADAD", xlab = "Strikeouts", main = "Histogram of Strikeouts")
hist(moneyball$TEAM_BATTING_HBP, col = "#DBCEAC", xlab = "Hit By Pitches", main = "Histogram of HBP")
boxplot(moneyball$TEAM_BATTING_BB, col = "#A71930", main = "Boxplot of Walks")
boxplot(moneyball$TEAM_BATTING_SO, col = "#09ADAD", main = "Boxplot of Strikeouts")
boxplot(moneyball$TEAM_BATTING_HBP, col = "#DBCEAC", main = "Boxplot of HBP")
par(mfrow=c(1,1))

# Stolen Bases and Caught Stealing
par(mfrow=c(2,2))
hist(moneyball$TEAM_BASERUN_SB, col = "#A71930", xlab = "Stolen Bases", main = "Histogram of Steals")
hist(moneyball$TEAM_BASERUN_CS, col = "#DBCEAC", xlab = "Caught Stealing", main = "Histogram of CS")
boxplot(moneyball$TEAM_BASERUN_SB, col = "#A71930", main = "Boxplot of Steals")
boxplot(moneyball$TEAM_BASERUN_CS, col = "#DBCEAC", main = "Boxplot of CS")
par(mfrow=c(1,1))

################ Pitching ############
# Hits and Home Runs
par(mfrow=c(2,1))
hist(moneyball$TEAM_PITCHING_HR, col = "#09ADAD", xlab = "Home Runs Against", main = "Histograms of HR Against")
boxplot(moneyball$TEAM_PITCHING_HR, col = "#09ADAD", main = "Boxplot of HR Against")
par(mfrow=c(1,1))

# If we remove outliers (we will impute later)
quicklook <- na.omit(data.frame(moneyball$TEAM_PITCHING_H))
quicklook <- quicklook[which(quicklook$moneyball.TEAM_PITCHING_H <= outlier),"moneyball.TEAM_PITCHING_H"]
hist(quicklook, col = "#A71930", xlab = "Hits Against", main = "Histogram of Hits Against")
# Becomes fairly normal, right skewed. 

# Walks and Strikeouts
par(mfrow=c(2,2))
hist(moneyball$TEAM_PITCHING_BB, col = "#A71930", xlab = "Walks Allowed", main = "Histogram of Walks Allowed")
hist(moneyball$TEAM_PITCHING_SO, col = "#DBCEAC", xlab = "Strikeouts", main = "Histograms of Strikeouts")
boxplot(moneyball$TEAM_PITCHING_BB, col = "#A71930", main = "Boxplot of Walks Allowed")
boxplot(moneyball$TEAM_PITCHING_SO, col = "#DBCEAC", main = "Boxplot of Strikeouts")
par(mfrow=c(1,1))

summary(moneyball$TEAM_PITCHING_BB)
outlier <- mean(moneyball$TEAM_PITCHING_BB)+IQR(moneyball$TEAM_PITCHING_BB)*1.5
quicklook <- na.omit(data.frame(moneyball$TEAM_PITCHING_BB))
quicklook <- quicklook[which(quicklook$moneyball.TEAM_PITCHING_BB <= outlier),1]
hist(quicklook, col = "#A71930", xlab = "Hits Against", main = "Histogram of Hits Against")
skewness(quicklook) # Normal.  Slightly left
kurtosis(quicklook) # Normal. Slightly leptokurtic

summary(moneyball$TEAM_PITCHING_SO)
quicklook <- na.omit(data.frame(moneyball$TEAM_PITCHING_SO))
outlier <- mean(quicklook$moneyball.TEAM_PITCHING_SO)+IQR(quicklook$moneyball.TEAM_PITCHING_SO)*1.5
quicklook <- quicklook[which(quicklook$moneyball.TEAM_PITCHING_SO <= outlier),1]
hist(quicklook, col = "#DBCEAC", xlab = "Strikeouts", main = "Histograms of Strikeouts")
skewness(quicklook) # Normal.  Slightly left
kurtosis(quicklook) 
# fairly normal. Metric indicates normality, but there's less height and more spread.

############## Fielding ###########
# Double Plays and Errors 
par(mfrow=c(2,2))
hist(moneyball$TEAM_FIELDING_DP, col = "#A71930", xlab = "Double Plays", main = "Histogram of Double Plays")
hist(moneyball$TEAM_FIELDING_E, col = "#09ADAD", xlab = "Errors Committed", main = "Histogram of Errors Committed")
boxplot(moneyball$TEAM_FIELDING_DP, col = "#A71930", main = "Boxplot of Double Plays")
boxplot(moneyball$TEAM_FIELDING_E, col = "#09ADAD", main = "Boxplot of Errors Committed")
par(mfrow=c(1,1))

# Errors are right skewed, with some outliers
summary(moneyball$TEAM_FIELDING_E)
outlier <- mean(moneyball$TEAM_FIELDING_E)+IQR(moneyball$TEAM_FIELDING_E)*1.5
nrow(moneyball[which(moneyball$TEAM_FIELDING_E >= outlier),])/nrow(moneyball)
# 13% are outliers, which is quite high.

######## Scatterplot Matrix ##########

panel.cor <- function(x, y, digits=2, prefix="", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits=digits)[1]
  txt <- paste(prefix, txt, sep="")
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_H,  moneyball$TEAM_FIELDING_DP, moneyball$TEAM_FIELDING_E))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_FIELDING_DP)
cor(quicklook$moneyball.TEAM_FIELDING_E, quicklook$moneyball.TEAM_FIELDING_DP)
cor(quicklook$moneyball.TEAM_PITCHING_H, quicklook$moneyball.TEAM_FIELDING_DP)

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_H, moneyball$TEAM_BATTING_2B))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_H)
# 0.39%
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_2B)
# .29%

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_3B, moneyball$TEAM_BATTING_HR))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_3B)
# .14

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_HR))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_HR)
# .176

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_BB, moneyball$TEAM_BATTING_SO))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_BB)
# .23
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BATTING_SO)
# -.03. Okay. strikeout by batter is expected to be Negative, but its negligible 

# Relationships between predictor variables
quicklook <- na.omit(data.frame(moneyball$TEAM_BATTING_H, moneyball$TEAM_BATTING_SO))
cor(quicklook$moneyball.TEAM_BATTING_H, quicklook$moneyball.TEAM_BATTING_SO)
# -.46. MAkes sense.

# SO has negative correlation with Hits, 3B, BB (walks by batters). Which makes sense
quicklook <- na.omit(data.frame(moneyball$TEAM_BATTING_HR, moneyball$TEAM_BATTING_SO))
cor(quicklook$moneyball.TEAM_BATTING_HR, quicklook$moneyball.TEAM_BATTING_SO)
# positive correlation with HR: .73%. And small positive with 2B (.16)

#Baserunning  Stats and Wins
pairs(~ moneyball$TARGET_WINS + moneyball$TEAM_BASERUN_CS + moneyball$TEAM_BASERUN_SB, lower.panel = panel.smooth)

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS,moneyball$TEAM_BASERUN_CS,moneyball$TEAM_BASERUN_SB))
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BASERUN_CS)
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_BASERUN_SB)
quicklook$RATIO_SB <- quicklook$moneyball.TEAM_BASERUN_SB/(quicklook$moneyball.TEAM_BASERUN_CS+quicklook$moneyball.TEAM_BASERUN_SB)
quicklook <- quicklook[which(quicklook$moneyball.TEAM_BASERUN_SB >0),]
cor(quicklook$moneyball.TARGET_WINS, quicklook$RATIO_SB)
# Ratio improves correlation slightly.

#Pitcher Stats and Wins
pairs(~ moneyball$TARGET_WINS + moneyball$TEAM_PITCHING_BB + moneyball$TEAM_PITCHING_H + 
        moneyball$TEAM_PITCHING_HR + moneyball$TEAM_PITCHING_SO, lower.panel = panel.smooth)

cor(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_H)
# -.11. # Makes sense. Hits allowed are negatively correlated with wins.

cor(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_HR)
# This shows a positive relationship (.19), why? 
# Maybe due to HR being negatiely related with hits. Allowing more HR but less hits.
# The more HR the more strike outs as well. 

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_SO, moneyball$TEAM_PITCHING_H))
cor(quicklook$moneyball.TEAM_PITCHING_H, quicklook$moneyball.TEAM_PITCHING_SO)
# .27 Hits allowed and strike outs.
cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_PITCHING_SO)
# Not much correlation. WE expect a positive correlation, slightly negative.
# This is due to hits allowed being correlated with strike outs. 

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_SO, moneyball$TEAM_PITCHING_HR))
quicklook$RATIO_PITCH_HR_SO <- (quicklook$moneyball.TEAM_PITCHING_HR/quicklook$moneyball.TEAM_PITCHING_SO)
quicklook <- quicklook[which(quicklook$moneyball.TEAM_PITCHING_SO > 0 & quicklook$moneyball.TEAM_PITCHING_HR > 0),]
cor(quicklook$moneyball.TARGET_WINS, quicklook$RATIO_PITCH_HR_SO)

cor(moneyball$TEAM_PITCHING_BB, moneyball$TEAM_PITCHING_HR)
# Walks allowed and HR allowed are correlated (.22), why?
cor(moneyball$TEAM_PITCHING_H, moneyball$TEAM_PITCHING_HR)
# Negative correlation between hits allowed and HR allows (-.14)
sum(moneyball$TEAM_PITCHING_HR)/sum(moneyball$TEAM_PITCHING_H)
# Since Hits are more common, this negative association might explain
# why HR allowed are positively correlated with WINS. 
quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_SO, moneyball$TEAM_PITCHING_HR))
cor(quicklook$moneyball.TEAM_PITCHING_SO, quicklook$moneyball.TEAM_PITCHING_HR)
# strikeouts by pitcher and HR allowed are correlated (.206)

cor(quicklook$moneyball.TARGET_WINS, quicklook$moneyball.TEAM_PITCHING_SO)
# HR are correlated with strike outs. strike outs are much more common (HR/SO is 13%)
# SO are negatively correlated with wins (-.078)
sum(quicklook$moneyball.TEAM_PITCHING_SO) 
sum(quicklook$moneyball.TEAM_PITCHING_HR)

# strike outs (good) and Hits allowed (negative) are positively correlated.
# Hits are twice as common. Which is why strike outs aren't positively correlated with wins
sum(quicklook$moneyball.TEAM_PITCHING_H)/sum(quicklook$moneyball.TEAM_PITCHING_SO)

quicklook <- na.omit(data.frame(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_HR,moneyball$TEAM_PITCHING_H))
quicklook$RATIO_HR_SO <- (quicklook$moneyball.TEAM_PITCHING_HR/quicklook$moneyball.TEAM_PITCHING_H)
quicklook <- quicklook[which(quicklook$moneyball.TEAM_PITCHING_H > 0),]
cor(quicklook$moneyball.TARGET_WINS, quicklook$RATIO_HR_H)

cor(quicklook$moneyball.TEAM_PITCHING_H, quicklook$moneyball.TEAM_PITCHING_BB)
# Hits allowed and walks allowed are positive correlated (.32)
# but diagram shows this is probably immaterial (due to outliers)

######################### Part 2: Data Preparation #####################
#Fix Missing Values Using Mean of All Seasons
moneyball$TEAM_BATTING_HBP[is.na(moneyball$TEAM_BATTING_HBP)] = mean(moneyball$TEAM_BATTING_HBP, na.rm = TRUE)
moneyball$TEAM_BASERUN_SB[is.na(moneyball$TEAM_BASERUN_SB)] = mean(moneyball$TEAM_BASERUN_SB, na.rm = TRUE)
moneyball$TEAM_BASERUN_CS[is.na(moneyball$TEAM_BASERUN_CS)] = mean(moneyball$TEAM_BASERUN_CS, na.rm = TRUE)
moneyball$TEAM_BASERUN_CS[moneyball$TEAM_BASERUN_CS == 0] = mean(moneyball$TEAM_BASERUN_CS, na.rm = TRUE)
moneyball$TEAM_FIELDING_DP[is.na(moneyball$TEAM_FIELDING_DP)] = median(moneyball$TEAM_FIELDING_DP, na.rm = TRUE)

moneyball$TEAM_BATTING_1B <- moneyball$TEAM_BATTING_H - moneyball$TEAM_BATTING_HR - moneyball$TEAM_BATTING_3B -
  moneyball$TEAM_BATTING_2B
moneyball[which(moneyball$TEAM_PITCHING_H < quantile(moneyball$TEAM_PITCHING_H,.01)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.01))
moneyball[which(moneyball$TEAM_PITCHING_H > quantile(moneyball$TEAM_PITCHING_H,.99)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.99))
moneyball[which(moneyball$TEAM_BATTING_3B < quantile(moneyball$TEAM_BATTING_3B,.01)),"TEAM_BATTING_3B"] = round(quantile(moneyball$TEAM_BATTING_3B,.01))
moneyball[which(moneyball$TEAM_BATTING_3B > quantile(moneyball$TEAM_BATTING_3B,.99)),"TEAM_BATTING_3B"] = round(quantile(moneyball$TEAM_BATTING_3B,.99))
moneyball[which(moneyball$TEAM_BATTING_HR < quantile(moneyball$TEAM_BATTING_HR,.01)),"TEAM_BATTING_HR"] = round(quantile(moneyball$TEAM_BATTING_HR,.01))
moneyball[which(moneyball$TEAM_BATTING_HR > quantile(moneyball$TEAM_BATTING_HR,.99)),"TEAM_BATTING_HR"] = round(quantile(moneyball$TEAM_BATTING_HR,.99))

moneyball[which(moneyball$TEAM_BASERUN_SB < quantile(moneyball$TEAM_BASERUN_SB,.01)),"TEAM_BASERUN_SB"] = round(quantile(moneyball$TEAM_BASERUN_SB,.01))
moneyball[which(moneyball$TEAM_BASERUN_SB > quantile(moneyball$TEAM_BASERUN_SB,.99)),"TEAM_BASERUN_SB"] = round(quantile(moneyball$TEAM_BASERUN_SB,.99))
moneyball[which(moneyball$TEAM_BASERUN_CS < quantile(moneyball$TEAM_BASERUN_CS,.01)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.01))
moneyball[which(moneyball$TEAM_BASERUN_CS > quantile(moneyball$TEAM_BASERUN_CS,.99)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.99))
moneyball$log_TEAM_BASERUN_SB <- log(moneyball$TEAM_BASERUN_SB)
moneyball$log_TEAM_BASERUN_CS <- log(moneyball$TEAM_BASERUN_CS)
moneyball$sqrt_TEAM_PITCHING_HR <- sqrt(moneyball$TEAM_PITCHING_HR)
moneyball$sqrt_TEAM_BATTING_HR <- sqrt(moneyball$TEAM_BATTING_HR)
moneyball$log_TEAM_BATTING_1B <- log(moneyball$TEAM_BATTING_1B)
moneyball$log_TEAM_BATTING_3B <- log(moneyball$TEAM_BATTING_3B)

moneyball[which(moneyball$TEAM_PITCHING_BB < quantile(moneyball$TEAM_PITCHING_BB,.01)),"TEAM_PITCHING_BB"] = round(quantile(moneyball$TEAM_PITCHING_BB,.01))
moneyball[which(moneyball$TEAM_PITCHING_BB > quantile(moneyball$TEAM_PITCHING_BB,.99)),"TEAM_PITCHING_BB"] = round(quantile(moneyball$TEAM_PITCHING_BB,.99))
moneyball[which(moneyball$TEAM_BATTING_BB < quantile(moneyball$TEAM_BATTING_BB,.01)),"TEAM_BATTING_BB"] = round(quantile(moneyball$TEAM_BATTING_BB,.01))
moneyball[which(moneyball$TEAM_BATTING_BB > quantile(moneyball$TEAM_BATTING_BB,.99)),"TEAM_BATTING_BB"] = round(quantile(moneyball$TEAM_BATTING_BB,.99))
moneyball$TEAM_FIELDING_E[(moneyball$TEAM_FIELDING_E > 500)] = 500
moneyball$TEAM_PITCHING_SO[is.na(moneyball$TEAM_PITCHING_SO)] = mean(moneyball$TEAM_PITCHING_SO, na.rm = TRUE)
moneyball[which(moneyball$TEAM_PITCHING_SO < quantile(moneyball$TEAM_PITCHING_SO,.01)),"TEAM_PITCHING_SO"] = round(quantile(moneyball$TEAM_PITCHING_SO,.01))
moneyball[which(moneyball$TEAM_PITCHING_SO > quantile(moneyball$TEAM_PITCHING_SO,.99)),"TEAM_PITCHING_SO"] = round(quantile(moneyball$TEAM_PITCHING_SO,.99))
summary(moneyball$TEAM_PITCHING_SO)
## Predict Batting SO ##
moneyball_BAT_SO_NNA <- moneyball[which(!is.na(moneyball$TEAM_BATTING_SO) & moneyball$TEAM_BATTING_SO != 0),]
moneyball_BAT_SO_NA <- moneyball[which(is.na(moneyball$TEAM_BATTING_SO) | moneyball$TEAM_BATTING_SO == 0),]

# We have to adjust the outliers before fitting.
moneyball_BAT_SO_NNA[which(moneyball_BAT_SO_NNA$TEAM_BATTING_SO < quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.01))
moneyball_BAT_SO_NNA[which(moneyball_BAT_SO_NNA$TEAM_BATTING_SO > quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.99))
BAT_SO_lmodel <- lm(TEAM_BATTING_SO ~ log_TEAM_BATTING_1B + log_TEAM_BATTING_3B +
                      sqrt_TEAM_BATTING_HR + TEAM_BATTING_BB + TEAM_PITCHING_SO, data = moneyball_BAT_SO_NNA)
summary(BAT_SO_lmodel)

BAT_SO_lmodel.test <- predict(BAT_SO_lmodel, newdata=moneyball_BAT_SO_NA);
moneyball$TEAM_BATTING_SO_predict <- moneyball$TEAM_BATTING_SO

moneyball[which(is.na(moneyball$TEAM_BATTING_SO) | moneyball$TEAM_BATTING_SO == 0), "TEAM_BATTING_SO_predict"] <- BAT_SO_lmodel.test
moneyball$TEAM_BATTING_SO[is.na(moneyball$TEAM_BATTING_SO)] = mean(moneyball$TEAM_BATTING_SO, na.rm = TRUE)
moneyball[which(moneyball$TEAM_BATTING_SO < quantile(moneyball$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.01))
moneyball[which(moneyball$TEAM_BATTING_SO > quantile(moneyball$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.99))
moneyball[which(moneyball$TEAM_BATTING_SO_predict < quantile(moneyball$TEAM_BATTING_SO_predict,.01)),"TEAM_BATTING_SO_predict"] = round(quantile(moneyball$TEAM_BATTING_SO_predict,.01))
moneyball[which(moneyball$TEAM_BATTING_SO_predict > quantile(moneyball$TEAM_BATTING_SO_predict,.99)),"TEAM_BATTING_SO_predict"] = round(quantile(moneyball$TEAM_BATTING_SO_predict,.99))
moneyball$TEAM_PITCHING_SO[moneyball$TEAM_PITCHING_SO == 0] = mean(moneyball$TEAM_PITCHING_SO)
moneyball[which(moneyball$TEAM_BATTING_SO_predict < quantile(moneyball$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.01))
moneyball[which(moneyball$TEAM_BATTING_SO_predict > quantile(moneyball$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.99))

# Additional transformations

# BATTING_BB
lmodel<- lm(TEAM_BATTING_BB ~ TARGET_WINS, data = moneyball)
boxcox(lmodel, plotit = TRUE)
y <- moneyball$TEAM_BATTING_BB
x <- moneyball$TARGET_WINS
bc <- boxcox(y ~ x)
m <- lm(y ~ moneyball$TEAM_BATTING_BB)
lambda_BB <- bc$x[which.max(bc$y)]
lambda_BB <- round(lambda_BB,0)

hist(moneyball$TEAM_BATTING_BB)
hist(moneyball$TEAM_BATTING_BB^lambda_BB)
cor(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_BB)
cor(moneyball$TARGET_WINS, moneyball$TEAM_BATTING_BB^lambda_BB)
moneyball$transform_TEAM_BATTING_BB <- moneyball$TEAM_BATTING_BB^round(lambda_BB)

#Check that na's are gone. 
summary(moneyball)

# Add variables
moneyball$SB_PCT <- moneyball$TEAM_BASERUN_SB/(1.0*moneyball$TEAM_BASERUN_SB+moneyball$TEAM_BASERUN_CS)
moneyball$AT_BAT_ADJ <- (moneyball$TEAM_PITCHING_SO+moneyball$TEAM_BATTING_H + moneyball$TEAM_BATTING_BB + moneyball$TEAM_BATTING_HBP)
moneyball$OBP_adj <- (moneyball$TEAM_BATTING_H + moneyball$TEAM_BATTING_BB + moneyball$TEAM_BATTING_HBP)/moneyball$AT_BAT_ADJ
moneyball$Score_Position <- (moneyball$TEAM_BATTING_1B + moneyball$TEAM_BATTING_BB + moneyball$TEAM_BATTING_HBP)*(moneyball$TEAM_BASERUN_SB/moneyball$AT_BAT_ADJ)/moneyball$SB_PCT + (moneyball$TEAM_BATTING_2B + moneyball$TEAM_BATTING_3B)/moneyball$AT_BAT_ADJ
moneyball$log_Score_Position <- log(moneyball$Score_Position)
moneyball$SLG <- (moneyball$TEAM_BATTING_1B + 2*moneyball$TEAM_BATTING_2B + 3*moneyball$TEAM_BATTING_3B + 4*moneyball$TEAM_BATTING_HR)/moneyball$AT_BAT_ADJ
moneyball$AVG <- moneyball$TEAM_PITCHING_H/moneyball$AT_BAT_ADJ
moneyball$HR_SO <- moneyball$TEAM_BATTING_HR/moneyball$TEAM_BATTING_SO

#Remove bad data from data set
moneyball2 <- moneyball
moneyball2[which(moneyball2$TARGET_WINS < 30),"TARGET_WINS"] = 30
moneyball2[which(moneyball2$TARGET_WINS > 120),"TARGET_WINS"] = 120 
moneyball2 <- subset(moneyball2, TEAM_PITCHING_H < 2000)
cor(moneyball$TARGET_WINS, moneyball$TEAM_PITCHING_H)
library(corrplot)
summary(moneyball2)
moneyball_corr <- cor(moneyball2[,1:10])
corrplot(moneyball_corr, method = "circle")

moneyball_corr <- cor(moneyball2[,c(2,3,18,4:17)])
corrplot(moneyball_corr, method = "circle")
moneyball_corr <- cor(moneyball2[,c(2,3,15,26:34)])
corrplot(moneyball_corr, method = "circle")
#################### Part 3: Model Creation ############################################


# Creating train/test split
set.seed(120)
moneyball2$u <- runif(n=dim(moneyball2)[1],min=0,max=1) #Creates random numbers that are uniformly distributed
trainset <- subset(moneyball2, u<0.70)
testset  <- subset(moneyball2, u>=0.70)

#Function for Mean Square Error Calculation
mse <- function(sm) 
  mean(sm$residuals^2)

# Stepwise Approach
stepwisemodel <- lm(formula = TARGET_WINS ~ log_TEAM_BATTING_1B + TEAM_BATTING_2B + TEAM_BATTING_3B + 
                      TEAM_BATTING_SO_predict + TEAM_PITCHING_SO + TEAM_BASERUN_SB + 
                      TEAM_PITCHING_HR + TEAM_BATTING_HR + TEAM_PITCHING_BB + transform_TEAM_BATTING_BB +
                      TEAM_PITCHING_H + TEAM_FIELDING_E +
                      Score_Position + TEAM_FIELDING_DP + 
                      SB_PCT + SLG, data = trainset)
stepwise <- stepAIC(stepwisemodel, direction = "both")
anova(stepwise)
summary(stepwise)
vif(stepwise)
sqrt(vif(stepwise)) > 2

# Model 2
model2 <- lm(TARGET_WINS ~ log_TEAM_BATTING_1B + log_TEAM_BATTING_3B + sqrt_TEAM_BATTING_HR + 
               TEAM_BASERUN_SB +
               TEAM_FIELDING_E + TEAM_FIELDING_DP + 
               TEAM_BATTING_SO_predict + TEAM_PITCHING_BB, data = trainset)
anova(model2)
summary(model2)

mse(model2)
model3 <- lm(TARGET_WINS ~ TEAM_BATTING_1B + TEAM_BATTING_3B + TEAM_BATTING_HR + 
               TEAM_BASERUN_SB + 
               TEAM_FIELDING_E + TEAM_FIELDING_DP + 
               TEAM_BATTING_SO_predict + TEAM_PITCHING_BB, data = trainset)
anova(model3)
summary(model3)
mse(model3)
vif(model3)
#plot(model3)

# Compare predictions to testset
LRmodel.test <- predict(model3, newdata=testset);

# Training Data
# Abs Pct Error
pct <- abs(model3$residuals)/trainset$TARGET_WINS; #Abs error/actual value so % error
MAPE <- mean(pct)  # Mean of % abs error.
MAPE # 10% +- error of final score

# Test Data
# Abs Pct Error
test.pct <- abs(testset$TARGET_WINS - LRmodel.test)/testset$TARGET_WINS

MAPE <- mean(test.pct)
MAPE


# Calculate RMS and MAE
rmse <- function(error)
{
  sqrt(mean(error^2))
}

actual <- testset$TARGET_WINS
predicted <- LRmodel.test
error_poly <- actual - predicted
rmse(error_poly)
summary(error_poly)
MAE <- mean(abs(error_poly))
MAE

# Cross-field validation to compare mean squared error scores
cv.lm(data = trainset, form.lm = formula(TARGET_WINS ~ log_TEAM_BATTING_1B +TEAM_BATTING_3B + TEAM_BATTING_HR + 
                                               TEAM_BASERUN_SB + 
                                               TEAM_FIELDING_E + TEAM_FIELDING_DP +  
                                               TEAM_BATTING_SO_predict + TEAM_PITCHING_BB), m=5, dots = FALSE, seed=26, plotit=TRUE, printit=TRUE)

######## Performance #######
AIC(stepwisemodel)
AIC(model2)
AIC(model3)
mse(stepwisemodel)
mse(model2)
mse(model3)
############ PCA ##############

moneyball2_reduced <- moneyball2[,c("TEAM_BATTING_1B", "TEAM_BATTING_2B","TEAM_BATTING_3B",
                                    "TEAM_BATTING_HR", "TEAM_BATTING_SO_predict", "TEAM_BASERUN_SB",
                                    "TEAM_BASERUN_CS", "TEAM_PITCHING_HR",
                                    "TEAM_PITCHING_BB", "TEAM_FIELDING_E", "TEAM_FIELDING_DP","TEAM_PITCHING_SO",  
                                    "transform_TEAM_BATTING_BB", "SB_PCT", "SLG",  "TEAM_PITCHING_H",
                                    "AVG", "Score_Position", "TARGET_WINS")]

moneyball2_reduced_2 <- moneyball2[,c("log_TEAM_BATTING_1B", "TEAM_BATTING_2B","log_TEAM_BATTING_3B",
                                      "sqrt_TEAM_BATTING_HR", "TEAM_BATTING_SO_predict", "log_TEAM_BASERUN_SB", 
                                      "log_TEAM_BASERUN_CS", "TEAM_PITCHING_HR",
                                      "TEAM_PITCHING_BB", "TEAM_FIELDING_E", "TEAM_FIELDING_DP","TEAM_PITCHING_SO",  
                                      "transform_TEAM_BATTING_BB", "SB_PCT", "SLG",  "TEAM_PITCHING_H",
                                      "AVG", "Score_Position", "TARGET_WINS")]

pca <- prcomp(moneyball2_reduced[,-19], center = TRUE,scale. = TRUE)
summary(pca)
str(pca)

pca2 <- prcomp(moneyball2_reduced_2[,-19], center = TRUE,scale. = TRUE)
summary(pca2)
str(pca)

returns.pca <- princomp(x=moneyball2_reduced[,-19],cor=TRUE) # Extracts principal components
returns.pca2 <- princomp(x=moneyball2_reduced_2[,-19],cor=TRUE) # Extracts principal components

return.scores <- as.data.frame(returns.pca$scores);
return.scores$TARGET_WINS <- moneyball2_reduced$TARGET_WINS;
return.scores$u <- runif(n=dim(return.scores)[1],min=0,max=1); # This is a uniform dist. of random scores 

return.scores2 <- as.data.frame(returns.pca2$scores);
return.scores2$TARGET_WINS <- moneyball2_reduced_2$TARGET_WINS;
return.scores2$u <- runif(n=dim(return.scores2)[1],min=0,max=1); # This is a uniform dist. of random scores 
# In order to split data.
head(return.scores)

# Split the data set into train and test data sets;
train.scores <- subset(return.scores,u<0.70);
test.scores <- subset(return.scores,u>=0.70);
train.scores2 <- subset(return.scores2,u<0.70);
test.scores2 <- subset(return.scores2,u>=0.70);

# Fit a linear regression model using the principal components;
pca1.lm <- lm(TARGET_WINS ~ Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6+Comp.7+Comp.8+Comp.9+Comp.10+Comp.11+Comp.12+Comp.13+Comp.14+Comp.15+Comp.16+Comp.17+Comp.18, data=train.scores);
pca2.lm <- lm(TARGET_WINS ~ Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6+Comp.7+Comp.8+Comp.9+Comp.10+Comp.11+Comp.12+Comp.13+Comp.14+Comp.15+Comp.16+Comp.17+Comp.18, data=train.scores2);
anova(pca1.lm)
summary(pca1.lm)
anova(pca2.lm)
summary(pca2.lm)

returns.pca
summary(returns.pca)
names(returns.pca)
pc.1 <- returns.pca$loadings[,1];
pc.2 <- returns.pca$loadings[,2];
names(pc.1)

# Plot the default scree plot;
plot(returns.pca, col = 'dodgerblue2', main = "Variance by Components") # How much variance is explained by each principal component.

# Make Scree Plot
scree.values <- (returns.pca$sdev^2)/sum(returns.pca$sdev^2);

plot(scree.values,xlab='Number of Components',ylab='',type='l',lwd=2)
points(scree.values,lwd=3,cex=1.5)
abline(h=0.025,lwd=1.5,col='red')
abline(v=8,lwd=1.5,col='red')
text(12,.05,'We see initial drop off',col='red')
title('Scree Plot')
# Another way of displaying the same results shown above. Looking for elbow.

# Make Proportion of Variance Explained
variance.values <- cumsum(returns.pca$sdev^2)/sum(returns.pca$sdev^2);

plot(variance.values,xlab='Number of Components',ylab='',type='l',lwd=2)
points(variance.values,lwd=2,cex=1.5)
abline(h=0.8,lwd=1.5,col='red')
abline(v=9,lwd=1.5,col='red')
text(13,0.5,'Drops off at 9 Principal Components',col='red')
title('Total Variance Explained Plot')

# Compute the Mean Absolute Error on the training sample;
pca1.lm$fitted.values[pca1.lm$fitted.values < 40] = 40
pca1.lm$fitted.values[pca1.lm$fitted.values > 115] = 115
pca1.mae.train <- mean(abs(train.scores$TARGET_WINS-pca1.lm$fitted.values));
mean(pca1.mae.train)
pca1.mse.train <- mean((train.scores$TARGET_WINS-pca1.lm$fitted.values)^2);
summary(pca1.lm$fitted.values)
pca1.Adj_R2.train <- summary(pca1.lm)$adj.r.squared
Adj_R2 <- 1 - (1 - summary(pca1.lm)$r.squared) * (nrow(train.scores) - 1)/(nrow(train.scores)-(length(pca1.lm$coefficients)-1)-1)

# Score the model out-of-sample and compute MAE;
pca1.test <- predict(pca1.lm,newdata=test.scores);

# RMSE and Adjusted R squared on Test Data set
R2 <- function(fitted, true){
  1 - (sum((true - fitted)^2)/sum((true - mean(true))^2))
}

pca1.R2 <- R2(pca1.test, test.scores$TARGET_WINS)
pca1.Adj_R2.test <- 1 - (1 - pca1.R2) * (nrow(test.scores) - 1)/(nrow(test.scores)-(length(pca1.lm$coefficients)-1)-1)

# MAE & MSE 
pca1.test[pca1.test < 40] = 40
pca1.test[pca1.test > 115] = 115
pca1.mae.test <- mean(abs(test.scores$TARGET_WINS-pca1.test));
pca1.mse.test <- mean((test.scores$TARGET_WINS-pca1.test)^2);
mean(pca1.mae.test)

# PCA 2
# Compute the Mean Absolute Error on the training sample;
pca2.lm$fitted.values[pca2.lm$fitted.values < 40] = 40
pca2.lm$fitted.values[pca2.lm$fitted.values > 115] = 115
pca2.mae.train <- mean(abs(train.scores2$TARGET_WINS-pca2.lm$fitted.values));
mean(pca2.mae.train)
pca2.mse.train <- mean((train.scores2$TARGET_WINS-pca2.lm$fitted.values)^2);
summary(pca2.lm$fitted.values)
pca2.Adj_R2.train <- summary(pca2.lm)$adj.r.squared
Adj_R2 <- 1 - (1 - summary(pca2.lm)$r.squared) * (nrow(train.scores2) - 1)/(nrow(train.scores2)-(length(pca2.lm$coefficients)-1)-1)

# Score the model out-of-sample and compute MAE;
pca2.test <- predict(pca2.lm,newdata=test.scores2);

pca2.R2 <- R2(pca2.test, test.scores2$TARGET_WINS)
pca2.Adj_R2.test <- 1 - (1 - pca2.R2) * (nrow(test.scores2) - 1)/(nrow(test.scores2)-(length(pca2.lm$coefficients)-1)-1)

# MAE & MSE 
pca2.test[pca2.test < 40] = 40
pca2.test[pca2.test > 115] = 115
pca2.mae.test <- mean(abs(test.scores2$TARGET_WINS-pca2.test));
pca2.mse.test <- mean((test.scores2$TARGET_WINS-pca2.test)^2);
mean(pca2.mae.test)


# Let's compare the PCA regression model with a 'raw' regression model;
# Create a train/test split of the returns data set to match the scores data set;
moneyball2$u <- return.scores$u;
train.returns <- subset(moneyball2,u<0.70);
test.returns <- subset(moneyball2,u>=0.70);
dim(train.returns)
dim(test.returns)
dim(train.returns)+dim(test.returns)
dim(moneyball2)

# Fit model2 on train data set and 'test' on test data;
model2 <- lm(TARGET_WINS ~ TEAM_BATTING_1B + TEAM_BATTING_3B + TEAM_BATTING_HR + 
               TEAM_BASERUN_SB + TEAM_BASERUN_CS + 
               TEAM_FIELDING_E + TEAM_FIELDING_DP +  
               TEAM_BATTING_SO + TEAM_PITCHING_BB, data=train.returns)
summary(model2)
model2$fitted.values[model2$fitted.values < 40] = 40
model2$fitted.values[model2$fitted.values > 115] = 115
model2.mae.train <- mean(abs(train.returns$TARGET_WINS-model2$fitted.values));
model2.mse.train <- mean((train.scores$TARGET_WINS-model2$fitted.values)^2);
 model2.Adj_R2.train <- summary(model2)$adj.r.squared

model2.test <- predict(model2,newdata=test.returns);
model2.test[model3.test < 40] = 40
model2.test[model3.test > 115] = 115
model2.mae.test <- mean(abs(test.returns$TARGET_WINS-model2.test));
model2.mse.test <- mean((test.returns$TARGET_WINS-model2.test)^2);
model2.R2 <- R2(model2.test, test.returns$TARGET_WINS)
model2.Adj_R2.test <- 1 - (1 - model2.R2) * (nrow(test.returns) - 1)/(nrow(test.returns)-(length(model2$coefficients)-1)-1)
anova(model2)
summary(model2)

# Fit model3 on train data set and 'test' on test data;
model3 <- lm(TARGET_WINS ~ TEAM_BATTING_1B + TEAM_BATTING_3B + TEAM_BATTING_HR + 
               TEAM_BASERUN_SB + TEAM_BASERUN_CS + 
               TEAM_FIELDING_E + TEAM_FIELDING_DP + 
               TEAM_BATTING_SO_predict + TEAM_PITCHING_BB, data=train.returns)
summary(model3)
model3$fitted.values[model3$fitted.values < 40] = 40
model3$fitted.values[model3$fitted.values > 115] = 115
model3.mae.train <- mean(abs(train.returns$TARGET_WINS-model3$fitted.values));
model3.mse.train <- mean((train.returns$TARGET_WINS-model3$fitted.values)^2);
model3.Adj_R2.train <- summary(model3)$adj.r.squared
model3.test <- predict(model3,newdata=test.returns);
model3.test[model3.test < 40] = 40
model3.test[model3.test > 115] = 115
model3.R2 <- R2(model3.test, test.returns$TARGET_WINS)
model3.Adj_R2.test <- 1 - (1 - model3.R2) * (nrow(test.returns) - 1)/(nrow(test.returns)-(length(model3$coefficients)-1)-1)
model3.mae.test <- mean(abs(test.returns$TARGET_WINS-model3.test));
model3.mse.test <- mean((test.scores$TARGET_WINS-model3.test)^2);
anova(model3)
summary(model3)

pca1.aic <- AIC(pca1.lm)
pca1.bic <- BIC(pca1.lm)
pca2.aic <- AIC(pca2.lm)
pca2.bic <- BIC(pca2.lm)
model2.aic <- AIC(model2)
model2.bic <- BIC(model2)
model3.aic <- AIC(model3)
model3.bic <- BIC(model3)

# Table of Metrics
rnames <- c("PCA1", "PCA2", "Model2", "Model3")
mse_mae_table <- matrix(list(pca1.mae.train, pca1.mae.test, pca1.mse.train, pca1.mse.test, pca1.Adj_R2.train, pca1.Adj_R2.test,  pca1.aic, pca1.bic, 
                             pca2.mae.train, pca2.mae.test, pca2.mse.train, pca2.mse.test, pca2.Adj_R2.train, pca2.Adj_R2.test,  pca2.aic, pca2.bic, 
                             model2.mae.train, model2.mae.test, model2.mse.train, model2.mse.test, model2.Adj_R2.train, model2.Adj_R2.test, model2.aic, model2.bic, 
                             model3.mae.train, model3.mae.test, model3.mse.train, model3.mse.test, model3.Adj_R2.train, model3.Adj_R2.test, model3.aic, model3.bic), ncol=8, byrow=TRUE)
colnames(mse_mae_table) <- c("MAE_Train","MAE_Test", "MSE_Train", "MSE_Test", "Adj_R2_Train", "Adj_R2_test", "AIC", "BIC")
mse_mae_table <- as.table(mse_mae_table)
mse_mae_table <- cbind(Row.Names = c(rnames), mse_mae_table)
mse_mae_table

#Applying best model to entire training set
#Fit a linear regression model using the principal components;
pca1.lm <- lm(TARGET_WINS ~ Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6+Comp.7+Comp.8+Comp.9+Comp.10+Comp.11+Comp.12+Comp.13+Comp.14+Comp.15+Comp.16+Comp.17, data=return.scores);
anova(pca1.lm)
summary(pca1.lm)

#################### Test Data ##########################

summary(moneyball_test)
#Fix Missing Values Using Mean of All Seasons
moneyball_test$TEAM_BASERUN_CS[is.na(moneyball_test$TEAM_BASERUN_CS)] = mean(moneyball$TEAM_BASERUN_CS, na.rm = TRUE)
moneyball_test$TEAM_BASERUN_CS[moneyball_test$TEAM_BASERUN_CS == 0] = mean(moneyball$TEAM_BASERUN_CS, na.rm = TRUE)
moneyball_test$TEAM_FIELDING_DP[is.na(moneyball_test$TEAM_FIELDING_DP)] = median(moneyball_test$TEAM_FIELDING_DP, na.rm = TRUE)
moneyball_test$TEAM_BASERUN_SB[is.na(moneyball_test$TEAM_BASERUN_SB)] = mean(moneyball$TEAM_BASERUN_SB, na.rm = TRUE)
moneyball_test[which(moneyball_test$TEAM_BASERUN_SB < quantile(moneyball$TEAM_BASERUN_SB,.01)),"TEAM_BASERUN_SB"] = round(quantile(moneyball$TEAM_BASERUN_SB,.01))
moneyball_test[which(moneyball_test$TEAM_BASERUN_SB > quantile(moneyball$TEAM_BASERUN_SB,.99)),"TEAM_BASERUN_SB"] = round(quantile(moneyball$TEAM_BASERUN_SB,.99))
moneyball_test[which(moneyball_test$TEAM_BASERUN_CS < quantile(moneyball$TEAM_BASERUN_CS,.01)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.01))
moneyball_test[which(moneyball_test$TEAM_BASERUN_CS > quantile(moneyball$TEAM_BASERUN_CS,.99)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.99))

moneyball_test[which(moneyball_test$TEAM_PITCHING_H < quantile(moneyball$TEAM_PITCHING_H,.01)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.01))
moneyball_test[which(moneyball_test$TEAM_PITCHING_H > quantile(moneyball$TEAM_PITCHING_H,.99)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.99))
moneyball_test[which(moneyball_test$TEAM_BATTING_3B < quantile(moneyball$TEAM_BATTING_3B,.01)),"TEAM_BATTING_3B"] = round(quantile(moneyball$TEAM_BATTING_3B,.01))
moneyball_test[which(moneyball_test$TEAM_BATTING_3B > quantile(moneyball$TEAM_BATTING_3B,.99)),"TEAM_BATTING_3B"] = round(quantile(moneyball$TEAM_BATTING_3B,.99))
moneyball_test[which(moneyball_test$TEAM_BATTING_HR < quantile(moneyball$TEAM_BATTING_HR,.01)),"TEAM_BATTING_HR"] = round(quantile(moneyball$TEAM_BATTING_HR,.01))
moneyball_test[which(moneyball_test$TEAM_BATTING_HR > quantile(moneyball$TEAM_BATTING_HR,.99)),"TEAM_BATTING_HR"] = round(quantile(moneyball$TEAM_BATTING_HR,.99))
moneyball_test[which(moneyball_test$TEAM_BASERUN_CS < quantile(moneyball$TEAM_BASERUN_CS,.01)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.01))
moneyball_test[which(moneyball_test$TEAM_BASERUN_CS > quantile(moneyball$TEAM_BASERUN_CS,.99)),"TEAM_BASERUN_CS"] = round(quantile(moneyball$TEAM_BASERUN_CS,.99))
moneyball_test$sqrt_TEAM_BATTING_HR <- sqrt(moneyball_test$TEAM_BATTING_HR)
moneyball_test$TEAM_BATTING_HBP[is.na(moneyball_test$TEAM_BATTING_HBP)] = mean(moneyball$TEAM_BATTING_HBP, na.rm = TRUE)

moneyball_test$TEAM_BATTING_1B <- moneyball_test$TEAM_BATTING_H - moneyball_test$TEAM_BATTING_HR - moneyball_test$TEAM_BATTING_3B -
  moneyball_test$TEAM_BATTING_2B
moneyball_test$log_TEAM_BATTING_1B <- log(moneyball_test$TEAM_BATTING_1B)
moneyball_test$log_TEAM_BATTING_3B <- log(moneyball_test$TEAM_BATTING_3B)

moneyball_test[which(moneyball_test$TEAM_PITCHING_BB < quantile(moneyball$TEAM_PITCHING_BB,.01)),"TEAM_PITCHING_BB"] = 240
moneyball_test[which(moneyball_test$TEAM_PITCHING_BB > quantile(moneyball$TEAM_PITCHING_BB,.99)),"TEAM_PITCHING_BB"] = 921 
moneyball_test[which(moneyball_test$TEAM_BATTING_BB < quantile(moneyball$TEAM_BATTING_BB,.01)),"TEAM_BATTING_BB"] = 80
moneyball_test[which(moneyball_test$TEAM_BATTING_BB > quantile(moneyball$TEAM_BATTING_BB,.99)),"TEAM_BATTING_BB"] = 753
moneyball_test$TEAM_FIELDING_E[(moneyball_test$TEAM_FIELDING_E > 500)] = 500
moneyball_test$TEAM_PITCHING_SO[is.na(moneyball_test$TEAM_PITCHING_SO)] = mean(moneyball$TEAM_PITCHING_SO, na.rm = TRUE)
moneyball_test[which(moneyball_test$TEAM_PITCHING_SO < quantile(moneyball$TEAM_PITCHING_SO,.01)),"TEAM_PITCHING_SO"] = round(quantile(moneyball$TEAM_PITCHING_SO,.01))
moneyball_test[which(moneyball_test$TEAM_PITCHING_SO > quantile(moneyball$TEAM_PITCHING_SO,.99)),"TEAM_PITCHING_SO"] = round(quantile(moneyball$TEAM_PITCHING_SO,.99))

# Predict Batting SO
moneyball_test_BAT_SO_NNA <- moneyball_test[which(!is.na(moneyball_test$TEAM_BATTING_SO) & moneyball_test$TEAM_BATTING_SO != 0),]
moneyball_test_BAT_SO_NA <- moneyball_test[which(is.na(moneyball_test$TEAM_BATTING_SO) | moneyball_test$TEAM_BATTING_SO == 0),]
moneyball_test_BAT_SO_NNA[which(moneyball_test_BAT_SO_NNA$TEAM_BATTING_SO < quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.01))
moneyball_test_BAT_SO_NNA[which(moneyball_test_BAT_SO_NNA$TEAM_BATTING_SO > quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball_BAT_SO_NNA$TEAM_BATTING_SO,.99))
summary(moneyball_test_BAT_SO_NNA$TEAM_BATTING_SO)
BAT_SO_lmodel.test <- predict(BAT_SO_lmodel, newdata=moneyball_test_BAT_SO_NA);
moneyball_test$TEAM_BATTING_SO_predict <- moneyball_test$TEAM_BATTING_SO
moneyball_test[which(is.na(moneyball_test$TEAM_BATTING_SO) | moneyball_test$TEAM_BATTING_SO == 0), "TEAM_BATTING_SO_predict"] <- BAT_SO_lmodel.test

moneyball_test[which(is.na(moneyball_test$TEAM_BATTING_SO) | moneyball_test$TEAM_BATTING_SO == 0), "TEAM_BATTING_SO_predict"] <- BAT_SO_lmodel.test
moneyball_test$TEAM_BATTING_SO[is.na(moneyball_test$TEAM_BATTING_SO)] = mean(moneyball$TEAM_BATTING_SO, na.rm = TRUE)
moneyball_test[which(moneyball_test$TEAM_BATTING_SO < quantile(moneyball$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.01))
moneyball_test[which(moneyball_test$TEAM_BATTING_SO > quantile(moneyball$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.99))
moneyball_test[which(moneyball_test$TEAM_BATTING_SO_predict < quantile(moneyball$TEAM_BATTING_SO_predict,.01)),"TEAM_BATTING_SO_predict"] = round(quantile(moneyball$TEAM_BATTING_SO_predict,.01))
moneyball_test[which(moneyball_test$TEAM_BATTING_SO_predict > quantile(moneyball$TEAM_BATTING_SO_predict,.99)),"TEAM_BATTING_SO_predict"] = round(quantile(moneyball$TEAM_BATTING_SO_predict,.99))
moneyball_test$TEAM_PITCHING_SO[moneyball_test$TEAM_PITCHING_SO == 0] = mean(moneyball$TEAM_PITCHING_SO)
moneyball_test[which(moneyball_test$TEAM_BATTING_SO_predict < quantile(moneyball$TEAM_BATTING_SO,.01)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.01))
moneyball_test[which(moneyball_test$TEAM_BATTING_SO_predict > quantile(moneyball$TEAM_BATTING_SO,.99)),"TEAM_BATTING_SO"] = round(quantile(moneyball$TEAM_BATTING_SO,.99))
moneyball_test[which(moneyball_test$TEAM_PITCHING_H < quantile(moneyball$TEAM_PITCHING_H,.01)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.01))
moneyball_test[which(moneyball_test$TEAM_PITCHING_H > quantile(moneyball$TEAM_PITCHING_H,.99)),"TEAM_PITCHING_H"] = round(quantile(moneyball$TEAM_PITCHING_H,.99))

# Additional transformations
moneyball_test$transform_TEAM_BATTING_BB <- moneyball_test$TEAM_BATTING_BB^round(lambda_BB)

#Check that na's are gone. 
summary(moneyball_test)

# Add variables
moneyball_test$SB_PCT <- moneyball_test$TEAM_BASERUN_SB/(1.0*moneyball_test$TEAM_BASERUN_SB+moneyball_test$TEAM_BASERUN_CS)
moneyball_test$AT_BAT_ADJ <- (moneyball_test$TEAM_PITCHING_SO+moneyball_test$TEAM_BATTING_H + moneyball_test$TEAM_BATTING_BB + moneyball_test$TEAM_BATTING_HBP)
moneyball_test$Score_Position <- (moneyball_test$TEAM_BATTING_1B + moneyball_test$TEAM_BATTING_BB + moneyball_test$TEAM_BATTING_HBP)*(moneyball_test$TEAM_BASERUN_SB/moneyball_test$AT_BAT_ADJ)/moneyball_test$SB_PCT + (moneyball_test$TEAM_BATTING_2B + moneyball_test$TEAM_BATTING_3B)/moneyball_test$AT_BAT_ADJ
moneyball_test$SLG <- (moneyball_test$TEAM_BATTING_1B + 2*moneyball_test$TEAM_BATTING_2B + 3*moneyball_test$TEAM_BATTING_3B + 4*moneyball_test$TEAM_BATTING_HR)/moneyball_test$AT_BAT_ADJ
moneyball_test$AVG <- moneyball_test$TEAM_PITCHING_H/moneyball_test$AT_BAT_ADJ
moneyball_test_reduced <- moneyball_test[,c("TEAM_BATTING_1B", "TEAM_BATTING_2B","TEAM_BATTING_3B",
                                            "TEAM_BATTING_HR", "TEAM_BATTING_SO_predict", "TEAM_BASERUN_SB",
                                            "TEAM_BASERUN_CS", "TEAM_PITCHING_HR",
                                            "TEAM_PITCHING_BB", "TEAM_FIELDING_E", "TEAM_FIELDING_DP","TEAM_PITCHING_SO",  
                                            "transform_TEAM_BATTING_BB", "SB_PCT", "SLG",  "TEAM_PITCHING_H",
                                            "AVG", "Score_Position")]
pca <- prcomp(moneyball_test_reduced, center = TRUE,scale. = TRUE)
components.pca <- princomp(x=moneyball_test_reduced,cor=TRUE) # Extracts principal components

moneyball_test_components <- as.data.frame(components.pca$scores);


# Stand Alone Scoring
LRmodel.test <- predict(pca1.lm, newdata=moneyball_test_components);
summary(pca1.lm)
moneyball_test$P_TARGET_WINS <- LRmodel.test
summary(moneyball_test$P_TARGET_WINS)
moneyball_test[which(moneyball_test$P_TARGET_WINS < 40),"P_TARGET_WINS"] = 40
moneyball_test[which(moneyball_test$P_TARGET_WINS > 115),"P_TARGET_WINS"] = 115
summary(moneyball_test$P_TARGET_WINS)

#subset of data set for the deliverable "Scored data file"
prediction <- moneyball_test[c("INDEX","P_TARGET_WINS")]

write.xlsx(prediction, file = "write.xlsx", sheetName = "Predictions", row.names = F)

