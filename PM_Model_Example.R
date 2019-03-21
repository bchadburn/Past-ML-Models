##Import needed packages
library(tidyr)
library(moments)
library(plyr)
require(corrplot)
require(ggplot2)
require(gridExtra)
require(car)
require(lattice)

setwd("D:/Piano Marvel Data/Raw Data/sasr_2018/users")
users_confirmed_student <- read.csv(file="users_confirmed_student.csv",head=TRUE,sep=",")
users_confirmed_teacher <- read.csv(file="users_confirmed_teacher.csv",head=TRUE,sep=",")
users_no_teacher <- read.csv(file="users_no_teacher.csv",head=TRUE,sep=",")
users_student <- read.csv(file="users_student.csv",head=TRUE,sep=",")

colnames(users_no_teacher) <- c("user_id", "names")
colnames(users_student) <- c("user_id", "names")
colnames(users_confirmed_student) <- c("user_id", "names")
users_student <- rbind(users_student, users_no_teacher, users_confirmed_student)
users_student <- unique(users_student)
users_student <- subset(users_student, !user_id %in% users_confirmed_teacher$X1)

setwd("D:/Piano Marvel Data/Raw Data/sasr_2018")
sessions <- read.csv(file="sasr_sessions.csv",head=TRUE,sep=",")
sdetails <- read.csv(file="sasr_session_details.csv",head=TRUE,sep=",")
pieces <- read.csv(file="sasr_pieces.csv",head=TRUE,sep=",")
not_teachers <- read.csv(file="users_not_teachers.csv",head=TRUE,sep=",")
pieces_moved <- read.csv(file="list_pieces_moved.csv", head=TRUE, sep = ",")
colnames(pieces_moved) <- c("piece_id", "level")

setwd("D:/Piano Marvel Data/Raw Data/SASR") 
past_sessions_details <- read.csv(file="SASR_session_details_15_to_17.csv",head=TRUE,sep=",")
past_sessions <- read.csv(file="SASR_sessions_15_to_17.csv",head=TRUE,sep=",")

# Get overview of the data
head(sessions)
names(sessions)

head(sdetails)
names(sdetails)

head(pieces)
tail(pieces)
names(pieces)

# Past sessions
head(past_sessions)
past_sessions <- past_sessions[,c(1,2,5,6,7)]
head(past_sessions)
names(past_sessions)
colnames(past_sessions) <- c("id", "created", "user_id", "score", "fail_count")
head(past_sessions_details)
past_sessions_details <- past_sessions_details[,c(1,2,4,5,6,7,8)]

# Evalute data. Check for missing values
summary(sessions)
nrow(pieces)
count(sessions$fail_count>3) # Spot check for issues

sessions <- subset(sessions, sessions$fail_count ==3)
past_sessions <-  subset(past_sessions, past_sessions$fail_count ==3)

# keep only sessions with 3 fail counts
sdetails <-  sdetails[which(sdetails$sasr_session_id %in% sessions$id),]
past_sessions_details <- past_sessions_details[which(past_sessions_details$sasr_session_id %in% past_sessions$id),]

# Convert difficulty levels to numeric
pieces$subLevel = sapply(as.character(pieces$subLevel), switch, "A" = "00", "B" = 20, "C" = 40, "D" = 60, "E" = 80,
                         USE.NAMES = F)
pieces$level <- paste(pieces$difficulty, pieces$subLevel, sep ="")
mode(sdetails$level)  # The level is now considered character due to "A"="00" so we need to convert it back to numeric
pieces$level <- as.numeric(pieces$level)
pieces <- pieces[ , c(1,2,5)]

# same thing to pieces in sdetails
sdetails$piece_sub_level = sapply(as.character(sdetails$piece_sub_level), switch, "A" = "00", "B" = 20, "C" = 40, "D" = 60, "E" = 80,
                                  USE.NAMES = F)
sdetails$level <- paste(sdetails$piece_level, sdetails$piece_sub_level, sep ="")
sdetails$level <- as.numeric(sdetails$level)
sdetails <- sdetails[ , -c(5,6)] # Dropped unneeded columns
colnames(sessions)[which(names(sessions) == "score")] <- "final_score"
sdetails <- merge(sdetails,sessions[ , c(1,3,4)],by.x="sasr_session_id", by.y ="id")

# Same thing to past_sessions_details
past_sessions_details$piece_sub_level = sapply(as.character(past_sessions_details$piece_sub_level), switch, "A" = "00", "B" = 20, "C" = 40, "D" = 60, "E" = 80,
                                               USE.NAMES = F)
past_sessions_details$level <- paste(past_sessions_details$piece_level, past_sessions_details$piece_sub_level, sep ="")
past_sessions_details$level <- as.numeric(past_sessions_details$level)
past_sessions_details <- past_sessions_details[ , -c(5,6)] # Dropped unneeded columns
colnames(past_sessions)[which(names(past_sessions) == "score")] <- "final_score"
past_sessions_details <- merge(past_sessions_details,past_sessions[ , c(1,3,4)],by.x="sasr_session_id", by.y ="id")
names(past_sessions_details)

######################################## Combining Current and Past Sessions ############################
past_sessions_details <- past_sessions_details[which(past_sessions_details$piece_id %in% pieces$id),]

# Checking if any pieces have changed within sdetails
check1 <- aggregate(sdetails$level, list(sdetails$piece_id),min)
check2 <- aggregate(sdetails$level, list(sdetails$piece_id),max)
check1$diff <- (check1$x-check2$x) # We find one piece that was changed in March. This is 'hard' coded for now.
helper <- sdetails[which(sdetails$piece_id == 5051 & sdetails$level == 980),]
sdetails <- sdetails[which(sdetails$piece_id != 5051),]
sdetails <- rbind(sdetails, helper)

helper_agg <- aggregate(sdetails$level, list(sdetails$piece_id), min)
colnames(helper_agg) <- c("piece_id", "level.x")
s_combine <- merge(past_sessions_details, helper_agg[,c("piece_id", "level.x"),], by = "piece_id", all.x = "TRUE")
s_combine$diff <- s_combine$level.x - s_combine$level
s_combine <- subset(s_combine, s_combine$diff  == 0)
s_combine <- s_combine[,1:8]
sdetails <- rbind(s_combine,sdetails)
check1 <- aggregate(sdetails$level, list(sdetails$piece_id),min)
check2 <- aggregate(sdetails$level, list(sdetails$piece_id),max)
sum(check1$x-check2$x) # We find one piece that was changed in March. This is 'hard' coded for now.

sdetails1 <- sdetails

########################## Remove sessions with teacher user_ids  ##############################
# We lose a lot of data here 60%+
remove_users <- c(105, 127,417, 317, 0) # Some other user_ids we know need to be removed (e.g. Aaron's)
sdetails <- subset(sdetails, !user_id %in% remove_users)
#sdetails <- subset(sdetails, user_id %in% users_student$user_id)
#sdetails <- subset(sdetails, user_id %in% users_confirmed_student)
#sdetails <- subset(sdetails, !user_id %in% users_confirmed_teacher)

########################## Remove exercises played more than once by same user ###################
sdetails <- sdetails[with(sdetails, order(created)), ]
sdetails$filter_first_play <- with(sdetails, paste0(user_id, piece_id))
sdetails <- sdetails[match(unique(sdetails$filter_first_play), sdetails$filter_first_play),]

########################## Removing initial sessions ##############################
sdetails <-  sdetails[with(sdetails, order(created)), ]
names(sdetails)
users_past <- unique(sdetails$user_id)
user_initial_test_current <- match(unique(sdetails$user_id),sdetails$user_id) # Row of 1st session by user
user_initial_test_current <- sdetails[user_initial_test_current,2] # Changed to list of sasr session

nrow(sdetails)
sdetails <- sdetails[which(sdetails$sasr_session_id %in% user_initial_test_current=="FALSE"),]  #Marks "True" all 1st sasr tests by user
nrow(sdetails) # We keep most sessions (94%).

############################### Data Wrangling ##################

# Prep
sdetails$change_fscore <- sdetails$final_score-sdetails$level # Create variable for change to final score from current level
session_initial <- sdetails[!duplicated(sdetails$sasr_session_id),c("sasr_session_id", "level")]
session_high <- aggregate(sdetails$level, list(sdetails$sasr_session_id),max)
session_initial$change <- session_high[,2] - session_initial[,2] 
colnames(session_initial) <- c('sasr_session_id', 'level', 'level_change')
sdetails <- merge(sdetails, session_initial[,c(1,3)], by = "sasr_session_id", all.x=TRUE)
sdetails$change_fscore <- sdetails$final_score - sdetails$level

# EDA
ggplot(sdetails, aes(x=level, y=score,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Score by level") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 100)),limits =c(0,1900)) +
  ylim(0,100)
# We can see low scores drag down levels 300-700,and 1700-1900

ggplot(sdetails, aes(x=level, y=change_fscore,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Score by level") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 100)),limits =c(0,1900)) +
  ylim(-200,400)
# At 700 we can see that a positive change in f_score. Around 1400-1500 we see a similar trend but not sure why. 

ggplot(sdetails, aes(x=level, y=level_change,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Score by level") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 100)),limits =c(0,1900)) +
  ylim(-600,1000)

# Add column for first level for each session (for reference)
sdetails <- sdetails[with(sdetails, order(created)), ]
first_exercise <- sdetails[match(unique(sdetails$sasr_session_id), sdetails$sasr_session_id),]
sdetails<- merge(sdetails, first_exercise[,c("sasr_session_id", "level")], by = "sasr_session_id", all.x=TRUE)
colnames(sdetails)[colnames(sdetails)=="level.x"] <- "level"
colnames(sdetails)[colnames(sdetails)=="level.y"] <- "first_level"

# Remove sessions with high changes
# high_changes <- subset(sdetails, sdetails$level_change > 2000 | sdetails$level_change < -2000)
# summary(sdetails$level_change)
# sdetails_mod <- subset(sdetails, !sdetails$sasr_session_id %in% high_changes$sasr_session_id)
# summary(sdetails_mod$level_change)

# We need to adjust what scores to keep in order to even out avg score by level.
sdetails_mod <- sdetails
upper_limit <- quantile(sdetails_mod$change_fscore, .95)
lower_limit <- quantile(sdetails_mod$change_fscore, .001)
#mean(sdetails[which(sdetails$level >= 1780), "change_fscore"])

sdetails_mod[which(sdetails_mod$change_fscore >= upper_limit), "change_fscore"] <- upper_limit
sdetails_mod[which(sdetails_mod$change_fscore <= lower_limit), "change_fscore"] <- lower_limit

sdetails_mod <- subset(sdetails_mod, sdetails_mod$score <= 100 & sdetails_mod$score >=40) 

quantile(sdetails$change_fscore,.99) # 1% of population. Could winzorize but its probably fine. 
quantile(sdetails$score,.1) # 50 and below make up about 10% of population. 0-50 mainly affect low count scores
# and 100-120 level. But they also bring down the overall average. When corrected the downward trend from 100-280 is steeper, 
# and the avg score trend from 300-1900 is more even (instead of an incline). Otherwise the trends between buckets are 
# very similar. 

# Same EDA as above with the changes
ggplot(sdetails_mod, aes(x=level, y=change_fscore,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Change_fscore by level: Score 40-100") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 100)),limits =c(0,1900)) +
  ylim(-200,400)
# At 700 we can see that a positive change in f_score. Around 1400-1500 we see a similar trend but not sure why. 

ggplot(sdetails_mod, aes(x = level, y = level_change, group = level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Score by level") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 100)),limits =c(0,1900)) +
  ylim(-600,1000)

sdetails_mod <- sdetails_mod1 # Create data copy

# Aggregating by piece and then by level (very similar to aggregating by level only)
bucket_review <- aggregate(sdetails_mod$score, list(sdetails_mod$piece_id),mean)
colnames(bucket_review) <- c("piece_id", "score")
bucket_review <- merge(bucket_review, pieces[,c(1,3)], by.x = "piece_id", by.y = "id", all.x=TRUE)
helper_change_fscore <- aggregate(sdetails_mod$change_fscore, list(sdetails_mod$piece_id),mean)
colnames(helper_change_fscore) <- c("piece_id", "change_fscore")
bucket_review <- merge(bucket_review, helper_change_fscore, by = "piece_id")
play_count <- aggregate(sdetails_mod$id, list(sdetails_mod$piece_id),length)
colnames(play_count) <- c("piece_id", "play_count")
bucket_review <- merge(bucket_review, play_count, by = "piece_id")
bucket_review <- subset(bucket_review, bucket_review$play_count >= 60)

bucket_review_level <- aggregate(bucket_review$score, list(bucket_review$level),mean)
colnames(bucket_review_level) <- c("level", "score")
helper_change <- aggregate(bucket_review$change_fscore, list(bucket_review$level),mean)
colnames(helper_change) <- c("level", "change_fscore")

bucket_review_level <- merge(bucket_review_level, helper_change)

summary(bucket_review_level$score)

ggplot(bucket_review_level, aes(x=level, y=score,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("All: Score by level") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 200)),limits =c(0,1900)) +
  ylim(50,100)

summary(bucket_review_level$change_fscore)

ggplot(bucket_review_level, aes(x=level, y=change_fscore,group=level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("All: change_fscore by level and score") +
  scale_x_continuous(breaks = round(seq(0, 1900, by = 200)),limits =c(0,1900)) +
  ylim(-100,150) 

######################################### EDA ######################################

# Quick correlations
cor(sdetails$final_score, sdetails$score) # We don't want these to be correlated
cor(sdetails$final_score, sdetails$level) # This of course makes sense, and is why we can expect a high R-squared.
cor(sdetails$change_fscore, sdetails$score) # After subtracting final score from current level. We do want to see correlation

cor(sdetails$level, sdetails$score) # negative correlation since 100-200 have higher scores and then it drops down.
cor(sdetails[which(sdetails$level>300 & sdetails$level<1600),'level'], sdetails[which(sdetails$level>300 & sdetails$level<1600),'score']) #close to 0

########################### Aggregatting scores to make predictions #################
pieces_reviewed <- aggregate(sdetails_mod$score, list(sdetails_mod$piece_id), mean)
final_score_avg <-  aggregate(sdetails_mod$final_score, list(sdetails_mod$piece_id), mean)
play_count <- aggregate(sdetails_mod$piece, list(sdetails_mod$piece_id), length)
pieces_reviewed <- merge(pieces_reviewed, pieces, by.x = "Group.1", by.y ="id", all.x=TRUE)
pieces_reviewed <- cbind(pieces_reviewed, final_score_avg, play_count)
pieces_reviewed <- pieces_reviewed[,c(1,2,4,6,8)]
colnames(pieces_reviewed)
summary(pieces_reviewed)
names(pieces_reviewed) <- c("piece_id", "score_avg", "current_level", "final_score_avg", "play_count")
# We see 10 NA's. These were pieces that were in the SASR that no longer show up in the list
# of pieces. We will delete these from the list. 
pieces_reviewed <- pieces_reviewed[!is.na(pieces_reviewed$current_level),]
pieces_reviewed$change_fscore <- pieces_reviewed$final_score_avg-pieces_reviewed$current_level # Create variable for change in final score based on score on exercise

cor(pieces_reviewed$current_level, pieces_reviewed$score_avg) 
cor(pieces_reviewed$final_score_avg, pieces_reviewed$current_level)
cor(pieces_reviewed$change_fscore, pieces_reviewed$score_avg) 
cor(pieces_reviewed$current_level, pieces_reviewed$change_fscore) 

ggplot(pieces_reviewed, aes(x=current_level, y=final_score_avg))+
  geom_point()+
  ylim(200,1200) +
  xlim(200,1200)
ggtitle("Change from Current level based on current level") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

avg_table <- aggregate(pieces_reviewed$score_avg, list(pieces_reviewed$current_level), mean)

ggplot(pieces_reviewed, aes(x=current_level, y=score_avg))+
  geom_point()+
  ylim(50,100) +
  scale_x_continuous(breaks = round(seq(100, 900, by = 200)),limits =c(100,900)) +
  ggtitle("Current level by score_avg") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(data = avg_table, aes(x=Group.1, y=x, colour="#000099"))

avg_table <- aggregate(pieces_reviewed$change_fscore, list(pieces_reviewed$current_level), mean)

ggplot(pieces_reviewed, aes(x=current_level, y=change_fscore))+
  geom_point()+
  ylim(0,150) +
  ggtitle("Change from Current level based on current level") +
  scale_x_continuous(breaks = round(seq(100, 900, by = 200)),limits =c(100,900)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_point(data = avg_table, aes(x=Group.1, y=x, colour="#000099"))

ggplot(pieces_reviewed, aes(x=current_level, y=change_fscore, group=current_level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Student: Change from Current level vs change_fscore") +
  geom_smooth(method = "lm") +
  ylim(0,125) +
  theme(plot.title = element_text(hjust = 0.5))  

ggplot(pieces_reviewed, aes(x=current_level, y=final_score_avg, group=current_level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ylim(200,600) +
  xlim(200,600) +
  ggtitle("Final Score Avg based on current level") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))  

low_level <- pieces_reviewed[which(pieces_reviewed$current_level <= 180),]
ggplot(low_level, aes(x=current_level, y=score_avg, group=current_level))+
  geom_boxplot(notch=FALSE, outlier.shape=NA, fill="red", alpha=0.2)+
  ggtitle("Change from Current level based on current level") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))  
# We can see that this isn't exactly linear, good but not as good as other levels. 

############################## Remove pieces with low play counts ################
pieces_reviewed <- pieces_reviewed[which(pieces_reviewed$play_count>=60),]
# Pieces are weighted equally, so we want to consider pieces with high play counts.

############################## Train/Test Split ###################################
# Create train/test split;
set.seed(124)
pieces_reviewed$u <- runif(n=dim(pieces_reviewed)[1],min=0,max=1) #Creates random numbers that are uniformly distributed
trainset <- subset(pieces_reviewed, u<0.70)
testset_final  <- subset(pieces_reviewed, u>=0.70)

########################### Building Model ####################

# Remove hymns
hymns <- c(2930,	2861,	2920,	2933,	2936,	2940,	2941,	2942,	2944,	2945,	2951,	2953,	2958,	2959,	2965,	3003,	3022,	
           4225,	4229,	4240,	4245,	4250,	4252,	4254,	4255,	4274,	4275,	4277,	4285,	4286,	4291,	4297,	4296,	4301,	
           4308,	4309,	4311,	4313,	4316,	4322,	4507,	4331,	4334,	4335,	4337,	4341,	4344,	4364,	4387,	4403,	4421,	
           4435,	4461,	4462,	4464,	4465,	4468,	4477,	4478,	4479,	4481,	4482,	4483,	4488,	4489,	4509,	4493,	4494,	
           4495,	4497,	4498,	4500,	4504,	4505,	4512,	4515,	4516,	4518,	4519,	4730,	4521,	4523,	4524,	4525,	4526,	
           4527,	4529,	4530,	4531,	4535,	4537,	4539,	4542,	4543,	4544,	4545,	4548,	4550,	4553,	4557,	4561,	4563,	
           4573,	4574,	4575,	4577,	4579,	4642,	4668,	4676,	4679,	4685,	4710,	4711,	4714,	4715,	4812,	4813,	4814,	
           5003,	5004,	5095,	5289,	5290,	5291,	5292,	5293,	5294,	5295,	5296,	6857,	10144,10205)

trainset <- subset(trainset, !piece_id %in% hymns)

subset_trainset1 <- trainset[which(trainset$current_level <= 180),]
LRmodel1 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset1)
anova(LRmodel1)
summary(LRmodel1)

subset_trainset2 <- trainset[which(trainset$current_level >= 200 & trainset$current_level <= 680),]
LRmodel2 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset2)
anova(LRmodel2)
summary(LRmodel2)

subset_trainset3 <- trainset[which(trainset$current_level >= 700 & trainset$current_level <= 880),]
LRmodel3 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset3)
anova(LRmodel3)
summary(LRmodel3)

par(mfrow=c(2,2))
plot(LRmodel3)

# subset_trainset4 <- trainset[which(trainset$current_level >= 900 & trainset$current_level <= 1900),]
# LRmodel4 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset4)
# anova(LRmodel4)
# summary(LRmodel4)

########################### Removing Outliers ############################
pred1 <- as.data.frame(predict(LRmodel1, subset_trainset1))
pred2 <- as.data.frame(predict(LRmodel2, subset_trainset2))
pred3 <- as.data.frame(predict(LRmodel3, subset_trainset3))
# pred4 <- as.data.frame(predict(LRmodel4, subset_trainset4))
summary(trainset_df$current_level)
colnames(pred2) =colnames(pred1)
colnames(pred3) = colnames(pred1)
#colnames(pred4) = colnames(pred1)

#pred <- rbind(pred1, pred2, pred3, pred4)
pred <- rbind(pred1, pred2, pred3)

names(pred)
pred <- rename(pred, c("predict(LRmodel1, subset_trainset1)" = "prd"))

#trainset_df <- rbind(subset_trainset1, subset_trainset2, subset_trainset3, subset_trainset4)
trainset_df <- rbind(subset_trainset1, subset_trainset2, subset_trainset3)

trainset_df$pred <- pred$prd
summary(pred)
trainset_df$res <- trainset_df$current_level - trainset_df$pred
trainset_df$absres <- abs(trainset_df$res)

summary(trainset_df$absres)
MAE <- mean(trainset_df$absres)
MAE

avg_table <- aggregate(trainset_df$res, list(trainset_df$current_level), mean)
ggplot(trainset_df, aes(x=score_avg, y=res))+
  geom_point()+
  ylim(-60,60) +
  ggtitle("Residuals based on score_avg") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(trainset_df, aes(x=current_level, y=res))+
  geom_point()+
  ylim(-75,75) +
  ggtitle("Residuals based on Current Level") +
  geom_smooth(method = "lm") +
  scale_x_continuous(breaks = round(seq(0, 1000, by = 200)),limits =c(0,1000)) +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(avg_table, aes(x=Group.1, y=x))+
  geom_point()+
  ylim(-60,60) +
  ggtitle("Residual Avg based on Current Level") +
  theme(plot.title = element_text(hjust = 0.5))

dffit_score1 <- dffits(LRmodel1)
dffit_score2 <- dffits(LRmodel2)
dffit_score3 <- dffits(LRmodel3)
#dffit_score4 <- dffits(LRmodel4)
trainset_df$dffit_score <- sample(100, size = nrow(trainset_df), replace = TRUE)
trainset_df[which(trainset_df$current_level <= 180),"dffit_score"] <- dffit_score1
trainset_df[which(trainset_df$current_level>=200 & trainset_df$current_level <= 680),"dffit_score"] <- dffit_score2
trainset_df[which(trainset_df$current_level>=700 & trainset_df$current_level <= 880),"dffit_score"] <- dffit_score3
#trainset_df[which(trainset_df$current_level>=900),"dffit_score"] <- dffit_score4

# Largeset absolute dffit_score
head(sort(abs(trainset_df$dffit_score),decreasing=TRUE),5)
trainset_df$absdf <- abs(trainset_df$dffit_score)

# Using equation to identify influential data points
p <- 16
n <- length(trainset_df$absdf)
dffits <- 2*(sqrt((p+1)/(n-p-1)))
nrow(trainset_df[trainset_df$dffit_score >= dffits,])/nrow(trainset_df) # influencial outliers makes small amount of data set.

# Review values with high dffits values
review <- trainset_df[which(trainset_df$absdf >= dffits),]
# Looking over these values, I'm not seeing common themes. I'll just remove these high values so they don't have impact
trainset_inf <- trainset_df[which(trainset_df$absdf < dffits),]

################################### Model with removed values #####################

subset_trainset1 <- trainset_inf[which(trainset_inf$current_level <= 180),]
LRmodel1 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset1)
anova(LRmodel1)
summary(LRmodel1)

par(mfrow=c(2,2))
plot(LRmodel1)

subset_trainset2 <- trainset_inf[which(trainset_inf$current_level >= 200 & trainset_inf$current_level <= 680),]
LRmodel2 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset2)
anova(LRmodel2)
summary(LRmodel2)

par(mfrow=c(2,2))
plot(LRmodel2)

subset_trainset3 <- trainset_inf[which(trainset_inf$current_level >= 700 & trainset_inf$current_level <= 880),]
LRmodel3 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset3)
anova(LRmodel3)
summary(LRmodel3)

par(mfrow=c(2,2))
plot(LRmodel3)
par(mfrow=c(1,1))
# subset_trainset4 <- trainset_inf[which(trainset_inf$current_level >= 900),]
# LRmodel4 <- lm(current_level ~ score_avg + final_score_avg, data=subset_trainset4)
# anova(LRmodel4)
# summary(LRmodel4)

cor(subset_trainset1$final_score_avg, subset_trainset1$score_avg)
cor(subset_trainset2$final_score_avg, subset_trainset2$score_avg)
cor(subset_trainset3$final_score_avg, subset_trainset3$score_avg)
#cor(subset_trainset4$final_score_avg, subset_trainset4$score_avg) 

# Check VIFs
library(car)
vif(LRmodel1)
vif(LRmodel2)
vif(LRmodel3)
#vif(LRmodel4)
##################################### Test data #######################

hymns <- c(2930,	2861,	2920,	2933,	2936,	2940,	2941,	2942,	2944,	2945,	2951,	2953,	2958,	2959,	2965,	3003,	3022,	
           4225,	4229,	4240,	4245,	4250,	4252,	4254,	4255,	4274,	4275,	4277,	4285,	4286,	4291,	4297,	4296,	4301,	
           4308,	4309,	4311,	4313,	4316,	4322,	4507,	4331,	4334,	4335,	4337,	4341,	4344,	4364,	4387,	4403,	4421,	
           4435,	4461,	4462,	4464,	4465,	4468,	4477,	4478,	4479,	4481,	4482,	4483,	4488,	4489,	4509,	4493,	4494,	
           4495,	4497,	4498,	4500,	4504,	4505,	4512,	4515,	4516,	4518,	4519,	4730,	4521,	4523,	4524,	4525,	4526,	
           4527,	4529,	4530,	4531,	4535,	4537,	4539,	4542,	4543,	4544,	4545,	4548,	4550,	4553,	4557,	4561,	4563,	
           4573,	4574,	4575,	4577,	4579,	4642,	4668,	4676,	4679,	4685,	4710,	4711,	4714,	4715,	4812,	4813,	4814,	
           5003,	5004,	5095,	5289,	5290,	5291,	5292,	5293,	5294,	5295,	5296,	6857,	10144,10205)

testset_final <- subset(testset_final, !piece_id %in% hymns)

testset_final1 <- testset_final[which(testset_final$current_level <= 180),]
testset_final2 <- testset_final[which(testset_final$current_level>=200 & testset_final$current_level <= 680),] 
testset_final3 <- testset_final[which(testset_final$current_level>=700 & testset_final$current_level <= 880),]
#testset_final4 <- testset_final[which(testset_final$current_level>=900),] 

LRmodel.test1 <- predict(LRmodel1,newdata=testset_final1);
LRmodel.test2 <- predict(LRmodel2,newdata=testset_final2);
LRmodel.test3 <- predict(LRmodel3,newdata=testset_final3);
#LRmodel.test4 <- predict(LRmodel4,newdata=testset_final4);

testset_final1$pred <- LRmodel.test1
testset_final2$pred <- LRmodel.test2
testset_final3$pred <- LRmodel.test3
#testset_final4$pred <- LRmodel.test4

testset_final$pred <- sample(100, size = nrow(testset_final), replace = TRUE)
testset_final[which(testset_final$current_level <= 180),"pred"] <-LRmodel.test1 
testset_final[which(testset_final$current_level>=200 & testset_final$current_level <= 680),"pred"] <-LRmodel.test2
testset_final[which(testset_final$current_level>=700 & testset_final$current_level <= 880),"pred"] <-LRmodel.test3
#testset_final[which(testset_final$current_level>=900),"pred"] <- LRmodel.test4

# Training Data
# Abs Pct Error
pct1 <- abs(LRmodel1$residuals)/subset_trainset1$current_level; #Abs error/actual value so % error
pct2 <- abs(LRmodel2$residuals)/subset_trainset2$current_level;
pct3 <- abs(LRmodel3$residuals)/subset_trainset3$current_level;
#pct4 <- abs(LRmodel4$residuals)/subset_trainset4$current_level;
# pct <- c(pct1,pct2, pct3, pct4)
pct <- c(pct1,pct2, pct3)
MAPE <- mean(pct)  # Mean of % abs error.
MAPE # % error of final score

# Test Data
# Abs Pct Error

test.pct <- abs(testset_final$current_level - testset_final$pred)/testset_final$current_level;

MAPE <- mean(test.pct)
MAPE

# Calculate RMS and MAE
rmse <- function(error)
{
  sqrt(mean(error^2))
}
summary(testset_final$current_level)

actual <- testset_final$current_level
predicted <- testset_final$pred
error_poly <- actual - predicted
summary(error_poly)
rmse(error_poly)
summary(abs(error_poly))

# Define change from prediction
testset_final$pred_change <- testset_final$pred -testset_final$current_level

# Evaluate error by play_count
testset_final$res <- abs(testset_final$current_level-testset_final$pred)
summary(testset_final$res)
length(which(abs(error_poly)>10))

ggplot(testset_final, aes(x=score_avg, y=change_fscore))+
  geom_point()+
  ggtitle("Change_fscore by score_avg") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(testset_final, aes(x=play_count, y=res))+
  geom_point()+
  ylim(0,40) +
  scale_x_continuous(breaks = round(seq(0, 800, by = 100)),limits =c(0,800)) +
  ggtitle("Error by Play_count") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(testset_final, aes(x=change_fscore, y=error_poly))+
  geom_point()+
  ylim(-40,40) +
  scale_x_continuous(breaks = round(seq(0, 160, by = 10)),limits =c(0,160)) +
  ggtitle("Error by change_fscore") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(testset_final, aes(x=change_fscore, y=res))+
  geom_point()+
  ylim(0,40) +
  scale_x_continuous(breaks = round(seq(0, 150, by = 10)),limits =c(0,150)) +
  ggtitle("Error by Play_count") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(testset_final, aes(x=score_avg, y=pred))+
  geom_point()+
  ylim(0,1000) +
  scale_x_continuous(breaks = round(seq(50, 100, by = 10)),limits =c(50,100)) +
  ggtitle("change by score") 
# We don't see too much of an incline. Around 80 our error starts approaching half a bucket in error. 

ggplot(testset_final, aes(x=current_level, y=error_poly))+
  geom_point()+
  ylim(-40,40) +
  scale_x_continuous(breaks = round(seq(0, 1400, by = 100)),limits =c(0,1400)) +
  ggtitle("Error by current level") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))
# This looks good, except 320-380 we see some issues. This makes sense due to the change in f_score trend and avg score trend at these levels

ggplot(testset_final, aes(x=score_avg, y=res))+
  geom_point()+
  ylim(0,40) +
  scale_x_continuous(breaks = round(seq(60, 100, by = 10)),limits =c(60,100)) +
  ggtitle("Error by score_avg") +
  theme(plot.title = element_text(hjust = 0.5))
# Homoskedasticity

ggplot(testset_final, aes(x=score_avg, y=error_poly))+
  geom_point()+
  ylim(-40,40) +
  scale_x_continuous(breaks = round(seq(60, 100, by = 10)),limits =c(60,100)) +
  ggtitle("Error by score_avg") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(testset_final, aes(x=score_avg, y=pred_change))+
  geom_point()+
  ylim(-100,100) +
  scale_x_continuous(breaks = round(seq(30, 100, by = 10)),limits =c(30,100)) +
  ggtitle("pred_change by score") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

hist(error_poly)
skewness(error_poly)
kurtosis(error_poly)
# Not normal. Slight left skew (prediction is low), too many 0-10 error. instead of -10 to 0. 

#################### Predict on all pieces  ########################

##################### Under 80 ######################
pieces_reviewed_low <- subset(pieces_reviewed, pieces_reviewed$play_count <80)
pieces_reviewed_low1 <- pieces_reviewed_low[which(pieces_reviewed_low$current_level <= 180),]
pieces_reviewed_low2 <- pieces_reviewed_low[which(pieces_reviewed_low$current_level>=200 & pieces_reviewed_low$current_level <= 680),] 
pieces_reviewed_low3 <- pieces_reviewed_low[which(pieces_reviewed_low$current_level>=700 & pieces_reviewed_low$current_level <= 880),]
#pieces_reviewed4 <- pieces_reviewed_low[which(pieces_reviewed_low$current_level>=900),] 

LRmodel.test1 <- predict(LRmodel1,newdata=pieces_reviewed_low1);
LRmodel.test2 <- predict(LRmodel2,newdata=pieces_reviewed_low2);
LRmodel.test3 <- predict(LRmodel3,newdata=pieces_reviewed_low3);
#LRmodel.test4 <- predict(LRmodel4,newdata=pieces_reviewed4);

pieces_reviewed_low1$pred <- LRmodel.test1
pieces_reviewed_low2$pred <- LRmodel.test2
pieces_reviewed_low3$pred <- LRmodel.test3
#pieces_reviewed4$pred <- LRmodel.test4


pieces_reviewed_low$pred <- sample(100, size = nrow(pieces_reviewed_low), replace = TRUE)
#pieces_reviewed_low[which(pieces_reviewed_low$current_level <= 180),"pred"] <-LRmodel.test1 
pieces_reviewed_low[which(pieces_reviewed_low$current_level>=200 & pieces_reviewed_low$current_level <= 680),"pred"] <-LRmodel.test2
pieces_reviewed_low[which(pieces_reviewed_low$current_level>=700 & pieces_reviewed_low$current_level <= 880),"pred"] <-LRmodel.test3
#pieces_reviewed_low[which(pieces_reviewed_low$current_level>=900),"pred"] <- LRmodel.test4

actual <- pieces_reviewed_low$current_level
predicted <- pieces_reviewed_low$pred
error_poly <- actual - predicted
summary(error_poly)
rmse(error_poly)
summary(abs(error_poly))
# Evaluate error by play_count
pieces_reviewed_low$res <- abs(pieces_reviewed_low$current_level-pieces_reviewed_low$pred)
pieces_reviewed_low$error <- (pieces_reviewed_low$current_level-pieces_reviewed_low$pred)
summary(pieces_reviewed_low$res)
cut_off_moved <- rmse(abs(error_poly))*2 #This is 2 st deviations of error + half a bucket
length(which(abs(error_poly)>cut_off_moved))
pieces_reviewed_low$new_level <- pieces_reviewed_low$current_level

pieces_reviewed_low[which(pieces_reviewed_low$error> cut_off_moved),"new_level"] <- pieces_reviewed_low[which(pieces_reviewed_low$error>  cut_off_moved),"pred"] 
pieces_reviewed_low[which(pieces_reviewed_low$error< -cut_off_moved),"new_level"] <- pieces_reviewed_low[which(pieces_reviewed_low$error<  -cut_off_moved),"pred"] 
length(which(abs(error_poly)>cut_off_moved))

pieces_reviewed_low$been_moved <- pieces_reviewed_low$piece_id %in% pieces_moved$piece_id

cut_off_new <- rmse(abs(error_poly))*1.5  #Error is 13. 400 is a 413.420. 413+10. 
nrow(pieces_reviewed_low[which(pieces_reviewed_low$been_moved == "FALSE" & pieces_reviewed_low$res>cut_off_new),])

pieces_reviewed_low[which(pieces_reviewed_low$been_moved == "FALSE" & pieces_reviewed_low$error> cut_off_new),"new_level"] <- pieces_reviewed_low[which(pieces_reviewed_low$been_moved == "FALSE" & pieces_reviewed_low$error> cut_off_new),"pred"]
pieces_reviewed_low[which(pieces_reviewed_low$been_moved == "FALSE" & pieces_reviewed_low$error < -cut_off_new),"new_level"] <- pieces_reviewed_low[which(pieces_reviewed_low$been_moved == "FALSE" & pieces_reviewed_low$error< -cut_off_new),"pred"]

pieces_reviewed_low$new_level <- round_any(pieces_reviewed_low$new_level, 20)
pieces_reviewed_low$change <- abs(pieces_reviewed_low$current_level - pieces_reviewed_low$new_level)>0
nrow(pieces_reviewed_low[which(pieces_reviewed_low$change == "TRUE"),])

################################## For scores with 80+ #############################
pieces_reviewed_mid <- subset(pieces_reviewed, pieces_reviewed$play_count >=80 & pieces_reviewed$play_count <120)
pieces_reviewed_mid1 <- pieces_reviewed_mid[which(pieces_reviewed_mid$current_level <= 180),]
pieces_reviewed_mid2 <- pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=200 & pieces_reviewed_mid$current_level <= 680),] 
pieces_reviewed_mid3 <- pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=700 & pieces_reviewed_mid$current_level <= 880),]
#pieces_reviewed4 <- pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=900),] 

LRmodel.test_mid1 <- predict(LRmodel1,newdata=pieces_reviewed_mid1);
LRmodel.test_mid2 <- predict(LRmodel2,newdata=pieces_reviewed_mid2);
LRmodel.test_mid3 <- predict(LRmodel3,newdata=pieces_reviewed_mid3);
#LRmodel.test_mid4 <- predict(LRmodel4,newdata=pieces_reviewed4);

pieces_reviewed_mid1$pred <- LRmodel.test_mid1
pieces_reviewed_mid2$pred <- LRmodel.test_mid2
pieces_reviewed_mid3$pred <- LRmodel.test_mid3
#pieces_reviewed4$pred <- LRmodel.test_mid4

pieces_reviewed_mid$pred <- sample(100, size = nrow(pieces_reviewed_mid), replace = TRUE)
pieces_reviewed_mid[which(pieces_reviewed_mid$current_level <= 180),"pred"] <-LRmodel.test_mid1 
pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=200 & pieces_reviewed_mid$current_level <= 680),"pred"] <-LRmodel.test_mid2
pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=700 & pieces_reviewed_mid$current_level <= 880),"pred"] <-LRmodel.test_mid3
#pieces_reviewed_mid[which(pieces_reviewed_mid$current_level>=900),"pred"] <- LRmodel.test_mid4

actual <- pieces_reviewed_mid$current_level
predicted <- pieces_reviewed_mid$pred
error_poly <- actual - predicted
summary(error_poly)
rmse(error_poly)
summary(abs(error_poly))

# Evaluate error by play_count
pieces_reviewed_mid$res <- abs(pieces_reviewed_mid$current_level-pieces_reviewed_mid$pred)
pieces_reviewed_mid$error <- (pieces_reviewed_mid$current_level-pieces_reviewed_mid$pred)
summary(pieces_reviewed_mid$res)
cut_off_mid_moved <- rmse(abs(error_poly))*2 #This is 2 st deviations of error
length(which(abs(error_poly)>cut_off_mid_moved))
pieces_reviewed_mid$new_level <- pieces_reviewed_mid$current_level

pieces_reviewed_mid[which(pieces_reviewed_mid$error> cut_off_mid_moved),"new_level"] <- pieces_reviewed_mid[which(pieces_reviewed_mid$error >  cut_off_mid_moved),"pred"] + 10
pieces_reviewed_mid[which(pieces_reviewed_mid$error< -cut_off_mid_moved),"new_level"] <- pieces_reviewed_mid[which(pieces_reviewed_mid$error <  -cut_off_mid_moved),"pred"] - 10
length(which(abs(error_poly)>cut_off_mid_moved))

pieces_reviewed_mid$been_moved <- pieces_reviewed_mid$piece_id %in% pieces_moved$piece_id

cut_off_mid_new <- rmse(abs(error_poly))*1.5
nrow(pieces_reviewed_mid[which(pieces_reviewed_mid$been_moved == "FALSE" & pieces_reviewed_mid$res>cut_off_mid_new),])

pieces_reviewed_mid[which(pieces_reviewed_mid$been_moved == "FALSE" & pieces_reviewed_mid$error > cut_off_mid_new),"new_level"] <- pieces_reviewed_mid[which(pieces_reviewed_mid$been_moved == "FALSE" & pieces_reviewed_mid$error> cut_off_mid_new),"pred"] + 10
pieces_reviewed_mid[which(pieces_reviewed_mid$been_moved == "FALSE" & pieces_reviewed_mid$error < -cut_off_mid_new),"new_level"] <- pieces_reviewed_mid[which(pieces_reviewed_mid$been_moved == "FALSE" & pieces_reviewed_mid$error< -cut_off_mid_new),"pred"] - 10

pieces_reviewed_mid$new_level <- round_any(pieces_reviewed_mid$new_level, 20)
pieces_reviewed_mid$change <- abs(pieces_reviewed_mid$current_level - pieces_reviewed_mid$new_level) > 0

################################## For scores with 120+ #############################
pieces_reviewed_high <- subset(pieces_reviewed, pieces_reviewed$play_count >=120)
pieces_reviewed_high1 <- pieces_reviewed_high[which(pieces_reviewed_high$current_level <= 180),]
pieces_reviewed_high2 <- pieces_reviewed_high[which(pieces_reviewed_high$current_level>=200 & pieces_reviewed_high$current_level <= 680),] 
pieces_reviewed_high3 <- pieces_reviewed_high[which(pieces_reviewed_high$current_level>=700 & pieces_reviewed_high$current_level <= 880),]
#pieces_reviewed4 <- pieces_reviewed_high[which(pieces_reviewed_high$current_level>=900),] 

LRmodel.test_high1 <- predict(LRmodel1,newdata=pieces_reviewed_high1);
LRmodel.test_high2 <- predict(LRmodel2,newdata=pieces_reviewed_high2);
LRmodel.test_high3 <- predict(LRmodel3,newdata=pieces_reviewed_high3);
#LRmodel.test_high4 <- predict(LRmodel4,newdata=pieces_reviewed4);

pieces_reviewed_high1$pred <- LRmodel.test_high1
pieces_reviewed_high2$pred <- LRmodel.test_high2
pieces_reviewed_high3$pred <- LRmodel.test_high3
#pieces_reviewed4$pred <- LRmodel.test_high4

pieces_reviewed_high$pred <- sample(100, size = nrow(pieces_reviewed_high), replace = TRUE)
pieces_reviewed_high[which(pieces_reviewed_high$current_level <= 180),"pred"] <-LRmodel.test_high1 
pieces_reviewed_high[which(pieces_reviewed_high$current_level>=200 & pieces_reviewed_high$current_level <= 680),"pred"] <-LRmodel.test_high2
pieces_reviewed_high[which(pieces_reviewed_high$current_level>=700 & pieces_reviewed_high$current_level <= 880),"pred"] <-LRmodel.test_high3
#pieces_reviewed_high[which(pieces_reviewed_high$current_level>=900),"pred"] <- LRmodel.test_high4

actual <- pieces_reviewed_high$current_level
predicted <- pieces_reviewed_high$pred
error_poly <- actual - predicted
summary(error_poly)
rmse(error_poly)
summary(abs(error_poly))

# Evaluate error by play_count
pieces_reviewed_high$res <- abs(pieces_reviewed_high$current_level-pieces_reviewed_high$pred)
pieces_reviewed_high$error <- (pieces_reviewed_high$current_level-pieces_reviewed_high$pred)
summary(pieces_reviewed_high$res)
cut_off_high_moved <- rmse(abs(error_poly))*2 #This is 2 st deviations of error + half a bucket
length(which(abs(error_poly)>cut_off_high_moved))
pieces_reviewed_high$new_level <- pieces_reviewed_high$current_level

pieces_reviewed_high[which(pieces_reviewed_high$error> cut_off_high_moved),"new_level"] <- pieces_reviewed_high[which(pieces_reviewed_high$error>  cut_off_high_moved),"pred"] + 10 
pieces_reviewed_high[which(pieces_reviewed_high$error< -cut_off_high_moved),"new_level"] <- pieces_reviewed_high[which(pieces_reviewed_high$error<  -cut_off_high_moved),"pred"] - 10
length(which(abs(error_poly)>cut_off_high_moved))

pieces_reviewed_high$been_moved <- pieces_reviewed_high$piece_id %in% pieces_moved$piece_id

cut_off_high_new <- rmse(abs(error_poly))*1.5 #This is 2 st deviations of error
nrow(pieces_reviewed_high[which(pieces_reviewed_high$been_moved == "FALSE" & pieces_reviewed_high$res>cut_off_high_new),])

pieces_reviewed_high[which(pieces_reviewed_high$been_moved == "FALSE" & pieces_reviewed_high$error> cut_off_high_new),"new_level"] <- pieces_reviewed_high[which(pieces_reviewed_high$been_moved == "FALSE" & pieces_reviewed_high$error> cut_off_high_new),"pred"] + 10
pieces_reviewed_high[which(pieces_reviewed_high$been_moved == "FALSE" & pieces_reviewed_high$error < -cut_off_high_new),"new_level"] <- pieces_reviewed_high[which(pieces_reviewed_high$been_moved == "FALSE" & pieces_reviewed_high$error< -cut_off_high_new),"pred"] - 10

pieces_reviewed_high$new_level <- round_any(pieces_reviewed_high$new_level, 20)
pieces_reviewed_high$change <- abs(pieces_reviewed_high$current_level - pieces_reviewed_high$new_level) > 0

pieces_reviewed_final <- rbind(pieces_reviewed_low, pieces_reviewed_mid, pieces_reviewed_high)
nrow(pieces_reviewed_final[which(pieces_reviewed_final$been_moved =="TRUE"),])
nrow(pieces_reviewed_final[which(pieces_reviewed_final$change == "TRUE"),])

moved_hymns <- subset(pieces_reviewed_final, piece_id %in% hymns & pieces_reviewed_final$change =="TRUE")
nrow(subset(moved_hymns, piece_id %in% hymns))
# of pieces to be moved that are hymns.

additional_changes <- pieces_reviewed_final[which(pieces_reviewed_final$been_moved == "TRUE" & pieces_reviewed_final$change == "TRUE"),]

# 2 pieces are being moved again but both are being moved up again after being moved up two buckets. 

# Manual review
avg_score_level <- aggregate(pieces_reviewed_final$score_avg, list(pieces_reviewed_final$current_level), mean)
colnames(avg_score_level) <- c("current_level", "score_avg_level")
pieces_reviewed_final <- merge(pieces_reviewed_final, avg_score_level, by = "current_level", all.x=TRUE)
pieces_reviewed_final$change_score_level <- pieces_reviewed_final$score_avg - pieces_reviewed_final$score_avg_level

avg_change_fscore_level <- aggregate(pieces_reviewed_final$change_fscore, list(pieces_reviewed_final$current_level), mean)
colnames(avg_change_fscore_level) <- c("current_level", "avg_change_fscore_level")
pieces_reviewed_final <- merge(pieces_reviewed_final, avg_change_fscore_level, by = "current_level", all.x=TRUE)
pieces_reviewed_final$change_fscore_level <- pieces_reviewed_final$change_fscore - pieces_reviewed_final$avg_change_fscore_level

ggplot(pieces_reviewed_final, aes(x=change_score_level, y=error))+
  geom_point()+
  ggtitle("pred_change by score") +
  geom_smooth(method = "lm") +
  theme(plot.title = element_text(hjust = 0.5))

cor(pieces_reviewed_final$change_score_level, pieces_reviewed_final$error)
hist(pieces_reviewed_final[which(pieces_reviewed_final$change == "TRUE"),"current_level"], main = "Level dist. of pieces moved")
