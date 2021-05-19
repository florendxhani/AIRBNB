rm(list = ls())    #delete objects
cat("\014")
install.packages("fastDummies")
install.packages("gridExtra")
library(plyr)
library(dplyr)
library(glmnet)
library(randomForest)
library(ggplot2)
library(fastDummies)
library(gridExtra)


#####################
### Cleaning Data ###
#####################

df <- read.csv("listing_data.csv")

### See Rows w/ NAs
sapply(df, function(x) sum(is.na(x)))

### Drop NAs
df <- df[complete.cases(df), ]

df %>% group_by(host_response_time) %>% tally()
df <- df[df$host_response_time != "unknown", ] 

df %>% group_by(experiences_offered) %>% tally()
df %>% group_by(host_response_time) %>% tally()
df %>% group_by(property_type) %>% tally()
df %>% group_by(room_type) %>% tally()
df %>% group_by(bed_type) %>% tally()

### Drop columns that do not provide differentiation and are just ids
df <- select(df, -country)
df <- select(df, -has_availability)
df <- select(df, -is_business_travel_ready)
df <- select(df, -id)
df <- select(df, -week_book)
df <- select(df, -month_book)
df <- select(df, -security)
df <- select(df, -requires_license)
df <- select(df, -neighbourhood_cleansed)

### Drop Other Rows
df <- subset(df, property_type!='Other')

### Change Values
#property type
df$property_type <- revalue(df$property_type, c("Hotel1"="Hotel"))
df$property_type <- revalue(df$property_type, c("Hotel2"="Hotel"))

#strict cancellation
df$strict_cancellation <- ifelse(grepl("strict", df$cancellation_policy, ignore.case = T), 1, 0)
df <- select(df, -cancellation_policy)

#real bed
df$real_bed <- ifelse(grepl("Real Bed", df$bed_type, ignore.case = T), 1, 0)
df <- select(df, -bed_type)

#experiences offered
df$offers_experiences <- ifelse(grepl("none", df$experiences_offered, ignore.case = T), 0, 1)
df <- select(df, -experiences_offered)

summary(df)

### Create Dummies and Drop First Val

#Check levels
df %>% group_by(host_response_time) %>% tally()
df %>% group_by(property_type) %>% tally()
df %>% group_by(room_type) %>% tally()

df <- dummy_cols(df, select_columns = c("host_response_time","property_type", "room_type"),remove_first_dummy = TRUE)
df <- select(df, -c("host_response_time","property_type", "room_type"))

colnames(df)

###Drop others
df <- select(df, -host_response_time_unknown)
df <- select(df, -property_type_Other)
df <- select(df, -availability_30)
df <- select(df, -availability_60)
df <- select(df, -availability_90)
df <- select(df, -availability_365)

colnames(df)
summary(df)

head(df)

nrow(df)
ncol(df)

list(colnames(df))

### Convert All Columns to Numeric
sapply(df, class)
df[] <- lapply(df, function(x) as.numeric(as.character(x)))
sapply(df, class)


##################################################
##### Variable Organization and Sample Params ####
##################################################

### Log Transformation
max(df$price)
min(df$price)
mean(df$price)

X <- select(df, !c(price))
X <- data.matrix(X)

y <- select(df, c(price))
y <- y + (1 - min (y))
y <- data.matrix(log(y))
y <- data.matrix(y)

hist(df$price,main = "Price without Transformation", xlab = "Price")
hist(y,main = "Price with Log Transformation", xlab = "Price")

n <- dim(X)[1]
p <- dim(X)[2]
n
p


### 80% train sample
sample <- round(n * 0.80)
n.train <- nrow(df[1:sample, ])
n.test <- nrow(df[(1 + sample):n, ])

### verify 80%
n.train/(n.train+n.test)

##########################
### Model Loops Number ###
##########################

loop = 100

#############
### Lists ###
#############

lasso.model.fill <- list(rep("Lasso", loop))
lasso.loop.time.fill <- list()
lasso.train.rsq.fill <- list()
lasso.test.rsq.fill <- list()
lasso.train.resid.fill <- list()
lasso.test.resid.fill <- list()
lasso.loop.fill <- list(1:loop)

elnet.model.fill <- list(rep("Elastic Net", loop))
elnet.loop.time.fill <- list()
elnet.train.rsq.fill <- list()
elnet.test.rsq.fill <- list()
elnet.train.resid.fill <- list()
elnet.test.resid.fill <- list()
elnet.loop.fill <- list(1:loop)

ridge.model.fill <- list(rep("Ridge", loop))
ridge.loop.time.fill <- list()
ridge.train.rsq.fill <- list()
ridge.test.rsq.fill <- list()
ridge.train.resid.fill <- list()
ridge.test.resid.fill <- list()
ridge.loop.fill <- list(1:loop)

randfor.model.fill <- list(rep("Random Forest", loop))
randfor.loop.time.fill <- list()
randfor.train.rsq.fill <- list()
randfor.test.rsq.fill <- list()
randfor.train.resid.fill <- list()
randfor.test.resid.fill <- list()
randfor.loop.fill <- list(1:loop)

for (i in c(1:loop)) {
  
  
  ###################################
  ##### RANDOM SAMPLE FORMATTING ####
  ###################################
  random <- sample(n)
  train <- random[1:n.train]
  test <- random[(1+n.train):n]
  X.train <- X[train, ]
  y.train <- y[train]
  X.test <- X[test, ]
  y.test <-y[test]
  
  #############
  ### LASSO ###
  #############
  
  lasso.start <- Sys.time()
  
  cv.fit.lasso <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit.lasso <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.lasso$lambda.min)
  y.train.pred.lasso <- predict(fit.lasso, newx = X.train, type = "response")
  y.test.pred.lasso <- predict(fit.lasso, newx = X.test, type = "response")
  rsq.train.lasso <- 1-mean((y.train - y.train.pred.lasso)^2)/mean((y - mean(y))^2)
  rsq.test.lasso <- 1-mean((y.test - y.test.pred.lasso)^2)/mean((y - mean(y))^2)
  resid.train.lasso <- as.vector(y.train - y.train.pred.lasso)
  resid.test.lasso <- as.vector(y.test - y.test.pred.lasso)
  
  lasso.end <- Sys.time()
  lasso.loop.time <- as.numeric(lasso.end - lasso.start)
  
  lasso.loop.time.fill <- append(lasso.loop.time.fill, lasso.loop.time)
  lasso.train.rsq.fill <- append(lasso.train.rsq.fill,rsq.train.lasso)
  lasso.test.rsq.fill <- append(lasso.test.rsq.fill,rsq.test.lasso)
  lasso.train.resid.fill <- append(lasso.train.resid.fill,rsq.train.lasso)
  lasso.test.resid.fill <- append(lasso.test.resid.fill,resid.test.lasso)
  
  
  #############
  ### ELNET ###
  #############  
  
  elnet.start <- Sys.time()
  
  cv.fit.elnet <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  fit.elnet <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.elnet$lambda.min)
  y.train.pred.elnet <- predict(fit.elnet, newx = X.train, type = "response")
  y.test.pred.elnet <- predict(fit.elnet, newx = X.test, type = "response")
  rsq.train.elnet <- 1-mean((y.train - y.train.pred.elnet)^2)/mean((y - mean(y))^2)
  rsq.test.elnet <- 1-mean((y.test - y.test.pred.elnet)^2)/mean((y - mean(y))^2)
  resid.train.elnet <- as.vector(y.train - y.train.pred.elnet)
  resid.test.elnet <- as.vector(y.test - y.test.pred.elnet)
  
  elnet.end <- Sys.time()
  elnet.loop.time <- as.numeric(elnet.end - elnet.start)
  
  elnet.loop.time.fill <- append(elnet.loop.time.fill, elnet.loop.time)
  elnet.train.rsq.fill <- append(elnet.train.rsq.fill,rsq.train.elnet)
  elnet.test.rsq.fill <- append(elnet.test.rsq.fill,rsq.test.elnet)
  elnet.train.resid.fill <- append(elnet.train.resid.fill,rsq.train.elnet)
  elnet.test.resid.fill <- append(elnet.test.resid.fill,resid.test.elnet)
 
  
  #############
  ### RIDGE ###
  #############
  
  ridge.start <- Sys.time()
  
  cv.fit.ridge <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit.ridge <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.ridge$lambda.min)
  y.train.pred.ridge <- predict(fit.ridge, newx = X.train, type = "response")
  y.test.pred.ridge <- predict(fit.ridge, newx = X.test, type = "response")
  rsq.train.ridge <- 1-mean((y.train - y.train.pred.ridge)^2)/mean((y - mean(y))^2)
  rsq.test.ridge <- 1-mean((y.test - y.test.pred.ridge)^2)/mean((y - mean(y))^2)
  resid.train.ridge <- as.vector(y.train - y.train.pred.ridge)
  resid.test.ridge <- as.vector(y.test - y.test.pred.ridge)
  
  ridge.end <- Sys.time()
  ridge.loop.time <- as.numeric(ridge.end - ridge.start)
  
  ridge.loop.time.fill <- append(ridge.loop.time.fill, ridge.loop.time)
  ridge.train.rsq.fill <- append(ridge.train.rsq.fill,rsq.train.ridge)
  ridge.test.rsq.fill <- append(ridge.test.rsq.fill,rsq.test.ridge)
  ridge.train.resid.fill <- append(ridge.train.resid.fill,rsq.train.ridge)
  ridge.test.resid.fill <- append(ridge.test.resid.fill,resid.test.ridge)
  
  
  #####################
  ### RANDOM FOREST ###
  #####################
  
  randfor.start <- Sys.time()
  
  fit.randfor <- randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.train.pred.randfor <- predict(fit.randfor, X.train)
  y.test.pred.randfor <- predict(fit.randfor, X.test)
  rsq.train.randfor <- 1-mean((y.train - y.train.pred.randfor)^2)/mean((y - mean(y))^2)
  rsq.test.randfor <- 1-mean((y.test - y.test.pred.randfor)^2)/mean((y - mean(y))^2)
  resid.train.randfor <- as.vector(y.train - y.train.pred.randfor)
  resid.test.randfor <- as.vector(y.test - y.test.pred.randfor)
  
  randfor.end <- Sys.time()
  randfor.loop.time <- as.numeric(randfor.end - randfor.start)
  
  randfor.loop.time.fill <- append(randfor.loop.time.fill, randfor.loop.time)
  randfor.train.rsq.fill <- append(randfor.train.rsq.fill,rsq.train.randfor)
  randfor.test.rsq.fill <- append(randfor.test.rsq.fill,rsq.test.randfor)
  randfor.train.resid.fill <- append(randfor.train.resid.fill,rsq.train.randfor)
  randfor.test.resid.fill <- append(randfor.test.resid.fill,resid.test.randfor)
  
}


###############################
### R2 Dataframes and Plots ###
###############################

train.fill <- list(rep("Train", loop))
test.fill <- list(rep("Test", loop))

lasso.train.r2.df <- data.frame(unlist(lasso.loop.fill),unlist(lasso.model.fill),unlist(lasso.train.rsq.fill),unlist(train.fill))
lasso.test.r2.df <- data.frame(unlist(lasso.loop.fill),unlist(lasso.model.fill),unlist(lasso.test.rsq.fill),unlist(test.fill))
lasso.train.r2.df <- setNames(lasso.train.r2.df, c("loop","model","R2","split"))
lasso.test.r2.df <- setNames(lasso.test.r2.df, c("loop","model","R2","split"))

elnet.train.r2.df <- data.frame(unlist(elnet.loop.fill),unlist(elnet.model.fill),unlist(elnet.train.rsq.fill),unlist(train.fill))
elnet.test.r2.df <- data.frame(unlist(elnet.loop.fill),unlist(elnet.model.fill),unlist(elnet.test.rsq.fill),unlist(test.fill))
elnet.train.r2.df <- setNames(elnet.train.r2.df, c("loop","model","R2","split"))
elnet.test.r2.df <- setNames(elnet.test.r2.df, c("loop","model","R2","split"))

ridge.train.r2.df <- data.frame(unlist(ridge.loop.fill),unlist(ridge.model.fill),unlist(ridge.train.rsq.fill),unlist(train.fill))
ridge.test.r2.df <- data.frame(unlist(ridge.loop.fill),unlist(ridge.model.fill),unlist(ridge.test.rsq.fill),unlist(test.fill))
ridge.train.r2.df <- setNames(ridge.train.r2.df, c("loop","model","R2","split"))
ridge.test.r2.df <- setNames(ridge.test.r2.df, c("loop","model","R2","split"))

randfor.train.r2.df <- data.frame(unlist(randfor.loop.fill),unlist(randfor.model.fill),unlist(randfor.train.rsq.fill),unlist(train.fill))
randfor.test.r2.df <- data.frame(unlist(randfor.loop.fill),unlist(randfor.model.fill),unlist(randfor.test.rsq.fill),unlist(test.fill))
randfor.train.r2.df <- setNames(randfor.train.r2.df, c("loop","model","R2","split"))
randfor.test.r2.df <- setNames(randfor.test.r2.df, c("loop","model","R2","split"))

r2.df <- rbind(lasso.train.r2.df,lasso.test.r2.df,elnet.train.r2.df,elnet.test.r2.df,ridge.train.r2.df,ridge.test.r2.df,randfor.train.r2.df,randfor.test.r2.df)

### write to csv just in case
write.csv(r2.df,"r2.df.csv")

level_order <- c('Elastic Net', 'Lasso', 'Ridge', 'Random Forest')
r2.plot <-ggplot(r2.df,aes(x=factor(model, level = level_order), y=R2, fill=model)) + geom_boxplot(fill="slateblue", alpha=0.2) + facet_wrap(~split) + xlab("Model")
r2.plot

### 90 CIs for Test R2
elnet_test_r2 <- filter(r2.df, split == "Test" & model == "Elastic Net")
lasso_test_r2 <- filter(r2.df, split == "Test" & model == "Lasso")
ridge_test_r2 <- filter(r2.df, split == "Test" & model == "Ridge")
randfor_test_r2 <- filter(r2.df, split == "Test" & model == "Random Forest")

quantile(elnet_test_r2$R2, c(.05, .95))
quantile(lasso_test_r2$R2, c(.05, .95))
quantile(ridge_test_r2$R2, c(.05, .95)) 
quantile(randfor_test_r2$R2, c(.05, .95))

### medians
elnet_train_r2 <- filter(r2.df, split == "Train" & model == "Elastic Net")
lasso_train_r2 <- filter(r2.df, split == "Train" & model == "Lasso")
ridge_train_r2 <- filter(r2.df, split == "Train" & model == "Ridge")
randfor_train_r2 <- filter(r2.df, split == "Train" & model == "Random Forest")

median(elnet_train_r2$R2)
median(elnet_test_r2$R2)

median(lasso_train_r2$R2)
median(lasso_test_r2$R2)

median(ridge_train_r2$R2)
median(ridge_test_r2$R2)

median(randfor_train_r2$R2)
median(randfor_test_r2$R2)

############################
### CV Samples and Plots ###
############################

random <- sample(n)
train <- random[1:n.train]
test <- random[(1+n.train):n]
X.train <- X[train, ]
y.train <- y[train]
X.test <- X[test, ]
y.test <-y[test]

### Lasso

lasso.start <- Sys.time()

cv.fit.lasso <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
fit.lasso <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.lasso$lambda.min)
y.train.pred.lasso <- predict(fit.lasso, newx = X.train, type = "response")
y.test.pred.lasso <- predict(fit.lasso, newx = X.test, type = "response")
rsq.train.lasso <- 1-mean((y.train - y.train.pred.lasso)^2)/mean((y - mean(y))^2)
rsq.test.lasso <- 1-mean((y.test - y.test.pred.lasso)^2)/mean((y - mean(y))^2)
resid.train.lasso <- as.vector(y.train - y.train.pred.lasso)
resid.test.lasso <- as.vector(y.test - y.test.pred.lasso)

lasso.end <- Sys.time()
lasso.time <- as.numeric(lasso.end - lasso.start)


### Elnet

elnet.start <- Sys.time()

cv.fit.elnet <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
fit.elnet <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.elnet$lambda.min)
y.train.pred.elnet <- predict(fit.elnet, newx = X.train, type = "response")
y.test.pred.elnet <- predict(fit.elnet, newx = X.test, type = "response")
rsq.train.elnet <- 1-mean((y.train - y.train.pred.elnet)^2)/mean((y - mean(y))^2)
rsq.test.elnet <- 1-mean((y.test - y.test.pred.elnet)^2)/mean((y - mean(y))^2)
resid.train.elnet <- as.vector(y.train - y.train.pred.elnet)
resid.test.elnet <- as.vector(y.test - y.test.pred.elnet)

elnet.end <- Sys.time()
elnet.time <- as.numeric(elnet.end - elnet.start)

### Ridge

ridge.start <- Sys.time()

cv.fit.ridge <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
fit.ridge <- glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.ridge$lambda.min)
y.train.pred.ridge <- predict(fit.ridge, newx = X.train, type = "response")
y.test.pred.ridge <- predict(fit.ridge, newx = X.test, type = "response")
rsq.train.ridge <- 1-mean((y.train - y.train.pred.ridge)^2)/mean((y - mean(y))^2)
rsq.test.ridge <- 1-mean((y.test - y.test.pred.ridge)^2)/mean((y - mean(y))^2)
resid.train.ridge <- as.vector(y.train - y.train.pred.ridge)
resid.test.ridge <- as.vector(y.test - y.test.pred.ridge)

ridge.end <- Sys.time()
ridge.time <- as.numeric(ridge.end - ridge.start)


### Random Forest

randfor.start <- Sys.time()

fit.randfor <- randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.train.pred.randfor <- predict(fit.randfor, X.train)
y.test.pred.randfor <- predict(fit.randfor, X.test)
rsq.train.randfor <- 1-mean((y.train - y.train.pred.randfor)^2)/mean((y - mean(y))^2)
rsq.test.randfor <- 1-mean((y.test - y.test.pred.randfor)^2)/mean((y - mean(y))^2)
resid.train.randfor <- as.vector(y.train - y.train.pred.randfor)
resid.test.randfor <- as.vector(y.test - y.test.pred.randfor)

randfor.end <- Sys.time()
randfor.time <- as.numeric(randfor.end - randfor.start)


#####################################
### Residual Dataframes and Plots ###
#####################################

lasso.train.res.df <- data.frame(unlist(resid.train.lasso))
lasso.test.res.df <- data.frame(unlist(resid.test.lasso))
lasso.train.res.df <- setNames(lasso.train.res.df, "residuals")
lasso.test.res.df <- setNames(lasso.test.res.df, "residuals")
lasso.train.res.df$model <- "Lasso"
lasso.test.res.df$model <- "Lasso"
lasso.train.res.df$split <- "Train"
lasso.test.res.df$split <- "Test"

elnet.train.res.df <- data.frame(unlist(resid.train.elnet))
elnet.test.res.df <- data.frame(unlist(resid.test.elnet))
elnet.train.res.df <- setNames(elnet.train.res.df, "residuals")
elnet.test.res.df <- setNames(elnet.test.res.df, "residuals")
elnet.train.res.df$model <- "Elastic Net"
elnet.test.res.df$model <- "Elastic Net"
elnet.train.res.df$split <- "Train"
elnet.test.res.df$split <- "Test"

ridge.train.res.df <- data.frame(unlist(resid.train.ridge))
ridge.test.res.df <- data.frame(unlist(resid.test.ridge))
ridge.train.res.df <- setNames(ridge.train.res.df, "residuals")
ridge.test.res.df <- setNames(ridge.test.res.df, "residuals")
ridge.train.res.df$model <- "Ridge"
ridge.test.res.df$model <- "Ridge"
ridge.train.res.df$split <- "Train"
ridge.test.res.df$split <- "Test"

randfor.train.res.df <- data.frame(unlist(resid.train.randfor))
randfor.test.res.df <- data.frame(unlist(resid.test.randfor))
randfor.train.res.df <- setNames(randfor.train.res.df, "residuals")
randfor.test.res.df <- setNames(randfor.test.res.df, "residuals")
randfor.train.res.df$model <- "Random Forest"
randfor.test.res.df$model <- "Random Forest"
randfor.train.res.df$split <- "Train"
randfor.test.res.df$split <- "Test"

res.df <- rbind(lasso.train.res.df,lasso.test.res.df,elnet.train.res.df,elnet.test.res.df,ridge.train.res.df,ridge.test.res.df,randfor.train.res.df,randfor.test.res.df)

### write to csv just in case
write.csv(res.df,"res.df.csv")

level_order <- c('Elastic Net', 'Lasso', 'Ridge', 'Random Forest')
res.df$split_f = factor(res.df$split, levels=c('Train','Test'))
res.plot <-ggplot(res.df,aes(x=factor(model, level = level_order), y=residuals, fill=model)) + geom_boxplot(fill="slateblue", alpha=0.2) + facet_grid(.~split_f) + xlab("Model")
res.plot

### create cv time data
cv.times <- c(lasso.time, elnet.time, ridge.time, randfor.time)
cv.models <- c("Lasso", "Elastic Net", "Ridge", "Random Forest")
cv.time.df <- data.frame(cv.models, cv.times)

### write to csv just in case
write.csv(cv.time.df,"cv.time.df.csv")

cv.time.df

#########################
### 10-Fold CV Curves ###
#########################

cv.fit.elnet <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
cv.fit.lasso <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
cv.fit.ridge <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

plot(cv.fit.elnet, sub = "Elastic Net")
plot(cv.fit.lasso, sub = "Lasso")
plot(cv.fit.ridge, sub = "Ridge")

cv.lambda.min <- c(cv.fit.elnet$lambda.min, cv.fit.lasso$lambda.min, cv.fit.ridge$lambda.min)
cv.models.min <- c("Elastic Net","Lasso","Ridge")
cv.min.df <- data.frame(cv.models.min, cv.lambda.min)
cv.min.df

### write to csv just in case
write.csv(cv.min.df,"cv.min.df.csv")




##############################
### Estimated Coefficients ###
##############################

lasso.full.start <- Sys.time()
cv.lasso    =      cv.glmnet(X, y, alpha=1, nfolds = 10)
lasso.fit   =      glmnet(X,y,alpha=1, lambda=cv.lasso$lambda)
lasso.full.end <- Sys.time()
lasso.full.time <- as.numeric(lasso.full.end - lasso.full.start)
beta.hat.lasso    =   lasso.fit$beta[ ,lasso.fit$lambda==cv.lasso$lambda.min]

ridge.full.start <- Sys.time()
cv.ridge    =      cv.glmnet(X, y, alpha=0, nfolds = 10)
ridge.fit   =      glmnet(X,y,alpha=0, lambda=cv.ridge$lambda)
ridge.full.end <- Sys.time()
ridge.full.time <- as.numeric(ridge.full.end - ridge.full.start)
beta.hat.ridge    =   ridge.fit$beta[ ,ridge.fit$lambda==cv.ridge$lambda.min]

elnet.full.start <- Sys.time()
cv.elnet    =      cv.glmnet(X, y,alpha=0.5, nfolds = 10)
df.elnet    =      cv.elnet$nzero[which.min(cv.elnet$cvm)]
elnet.fit   =      glmnet(X,y,alpha=0.5, lambda=cv.elnet$lambda)
elnet.full.end <- Sys.time()
elnet.full.time <- as.numeric(elnet.full.end - elnet.full.start)
beta.hat.elnet    =   elnet.fit$beta[ ,elnet.fit$lambda==cv.elnet$lambda.min]

rf.full.start <- Sys.time()
rf.fit = randomForest(X, as.vector(y), mtry = sqrt(p), importance = TRUE)
beta.hat.rf = data.frame(names(X[1,]), as.vector(rf.fit$importance[,1]))
rf.full.end <- Sys.time()
rf.full.time <- as.numeric(rf.full.end - rf.full.start)

### create cv time data
cv.full.times <- c(lasso.full.time, elnet.full.time, ridge.full.time, rf.full.time)
cv.models <- c("Lasso", "Elastic Net", "Ridge", "Random Forest")
cv.full.time.df <- data.frame(cv.models, cv.full.times)

### write to csv just in case
write.csv(cv.full.time.df,"cv.full.time.df.csv")

s = apply(X, 2, sd)

betaS.en.norm               =     data.frame(c(1:p), as.vector(beta.hat.elnet) * s)
colnames(betaS.en.norm)     =     c( "feature", "value")

betaS.ls.norm               =     data.frame(c(1:p), as.vector(beta.hat.lasso) * s)
colnames(betaS.ls.norm)     =     c( "feature", "value")

betaS.rg.norm               =     data.frame(c(1:p), as.vector(beta.hat.ridge) * s)
colnames(betaS.rg.norm)     =     c( "feature", "value")

### do not standardize random forest
betaS.rf               =     data.frame(c(1:p), as.vector(beta.hat.rf))
colnames(betaS.rf)     =     c( "feature", "Irrel","value")
betaS.rf = select(betaS.rf, -Irrel)
betaS.rf


lsPlot =  ggplot(betaS.ls.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    


enPlot =  ggplot(betaS.en.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")  

rgPlot =  ggplot(betaS.rg.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")  


grid.arrange(enPlot, lsPlot, rgPlot, rfPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.ls.norm$feature     =  factor(betaS.ls.norm$feature, levels = betaS.en.norm$feature[order(betaS.en.norm$value, decreasing = TRUE)])
betaS.en.norm$feature     =  factor(betaS.en.norm$feature, levels = betaS.en.norm$feature[order(betaS.en.norm$value, decreasing = TRUE)])
betaS.rg.norm$feature     =  factor(betaS.rg.norm$feature, levels = betaS.en.norm$feature[order(betaS.en.norm$value, decreasing = TRUE)])
betaS.rf$feature          =  factor(betaS.rf$feature, levels = betaS.en.norm$feature[order(betaS.en.norm$value, decreasing = TRUE)])


lsPlot =  ggplot(betaS.ls.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+ ggtitle("Lasso")+
  theme(axis.title.x=element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

enPlot =  ggplot(betaS.en.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+ ggtitle("Elastic Net")+
  theme(axis.title.x=element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

rgPlot =  ggplot(betaS.rg.norm, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+ ggtitle("Ridge")+
  theme(axis.title.x=element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") + ggtitle("Random Forest")+
  theme(axis.title.x=element_blank()) +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(enPlot, lsPlot, rgPlot, rfPlot, nrow = 4)

ncol(X)
colnames(X)

#6 - accommodates
#8 - bedrooms
#11 - cleaning fee

#42 - room_type_Private room"
#26 - distance
