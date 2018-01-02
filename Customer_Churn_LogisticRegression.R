#Logistic Regression to find out customer churn rate for a bank.
#We will use CAP curve to asses our model.
#Confusion Matrix will help in finding out the sensitivity and specificity of our model.

#Read the data
cust_df <- read.csv("G:\\Data Science A-Z\\Modelling\\Churn-Modelling with test data appended.csv")
head(cust_df)

#Let's create dummy variable for categorical variable: Geography and Gender

Gender_male <- ifelse(cust_df$Gender == 'Male',1,0)
Gender_female <- ifelse(cust_df$Gender == 'Female',1,0)

geo_Frc <- ifelse(cust_df$Geography == 'France',1,0)
geo_Spa <- ifelse(cust_df$Geography == 'Spain',1,0)
geo_Ger <- ifelse(cust_df$Geography == 'Germany',1,0)

#We don't need customerID and surname
#Among the categorical variables, we don't need to include all.
#We can exclulde one dummy variable from each category.
#As by inluding all but one dummy variable from any given category, we are including info for the entire category.
#For example, 
my_df <- cbind(cust_df[,c(4,7,8,9,10,11,12,13,14)],Gender_female,geo_Frc,geo_Ger)

#Let's build our model
#1st iteration
#With all independent variables
model1 <- glm(Exited ~ .,data=my_df[c(1:10000),],family = binomial(link = "logit"))
summary(model1)

#We need to remove the most insignificant variable through backward elimination.
#Here, geo_FRC is the least informative varibale.
#Let's remove that and rebuild our model.

model2 <- glm(Exited ~ CreditScore+Age+Tenure+Balance+NumOfProducts+HasCrCard+IsActiveMember+EstimatedSalary+Gender_female+geo_Ger, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model2)

#Here, HasCrCard is the least informative and non-significant varibale.
#Let's remove that and rebuild our model.

model3 <- glm(Exited ~ CreditScore+Age+Tenure+Balance+NumOfProducts+IsActiveMember+EstimatedSalary+Gender_female+geo_Ger, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model3)


#Here, EstimatedSalary is the least informative and non-significant varibale.
#Let's remove that and rebuild our model.

model4 <- glm(Exited ~ CreditScore+Age+Tenure+Balance+NumOfProducts+IsActiveMember+Gender_female+geo_Ger, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model4)

#Here, Tenure is the least informative and non-significant varibale.
#Let's remove that and rebuild our model.

model5 <- glm(Exited ~ CreditScore+Age+Balance+NumOfProducts+IsActiveMember+Gender_female+geo_Ger, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model5)

#IF we compare model4 and model5. we can see that even though the variable Tenure is not that significant, it should remain in the model.
#Cause AIC of model5 is higher than model4 and model4 has a low residual deviance as compared to model5.
#Also, business knowledge dictates that Tenure/no of years the customer has been with a bank is crucial. As older customers tend to build
#a relation with the bank and would want to stick with the same bank.
#Therefore, in our case, the bank is obviously doing something wrong in terms of building/maintaining that relationship.
#Else, tenure would have been more significant in our model.

#We select model4 for further analysis.

#Next step is to see if any variable needs transformation.
#Balance can be transformed using log transformation. This would provide for a more consistent effect/analysis on an unit increase 
#in balance.
Bal_log <- ifelse(my_df$Balance ==0,0,log10(my_df$Balance))
my_df <- cbind(my_df[,c(1,2,3,5,6,7,8,9,10,11,12)],Bal_log)
model6 <- glm(Exited ~ CreditScore+Age+Tenure+Bal_log+NumOfProducts+IsActiveMember+Gender_female+geo_Ger, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model6, corr=T)

#If we notice the variables in the model, we can see Balance and Age might have some connection among them. People who are older tend to 
#have higher balance in their account. There are also some people who like to save money and accumulate higher balance in tehir account.
#As of now, we are treating them equally. However, we need to separate those two types of people and other similar effects.

#One such way is to divide balance over age


WealthAccum <- cust_df$Balance/cust_df$Age
my_df <- cbind(my_df,WealthAccum)
model7 <- glm(Exited ~ CreditScore+Age+Tenure+Bal_log+NumOfProducts+IsActiveMember+Gender_female+geo_Ger+WealthAccum, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model7, corr=T)

#We can see that this new variable is not so significant. But, it may also be possible due to multi-collinearity between Bal_log, Age and WealthAccum
#WealthAccum and Bal_log have high correlation. Let's see the correlation plot.

#Plot Correlation
mycor <- cor(my_df[c(1:10000),c(1,2,3,4,6,9,11,12,13)])
require(corrplot)
corrplot(mycor,method = "number")
vif(model7)

#So we can see clear correlation between wealth accumulation and balance log.
#Let's go ahead and add another variable by taking log of wealth accumulation to see how it fits in the model.

WealthAccum_log <- ifelse(cust_df$Balance ==0,0,log10(cust_df$Balance/cust_df$Age))
my_df <- cbind(my_df,WealthAccum_log)
model8 <- glm(Exited ~ CreditScore+Age+Tenure+Bal_log+NumOfProducts+IsActiveMember+Gender_female+geo_Ger+WealthAccum_log, data = my_df[c(1:10000),], family = binomial(link="logit"))
summary(model8, corr=T)

#This model performs better than all previous models as the AIC in this model is lowest and residual deviance is also lower.
#All variables are significant too
#But, what about correlation? Seems wealth accumulation and Bal-log have high correlation with Age. Let's check vif.

vif(model8)

#vif for wealth accumulation and balance log is way beyond the threshold.
#Therefore, we can establish that wealth accumulation can be left out of the model.

#We can go back and select model6 to proceed.
#Let's revisit the summary in model 6.

summary(model6, corr = T)
vif(model6)
exp(coef(model6))
#Correlation/vif figures look under control.

#Predict on the same data set to build a CAP curve and test accuracy
predicted_train <- predict(model6, newdata = my_df[c(1:10000),],type='response')
head(predicted_train)

#Write the actual exited vs predicted probablities to a csv for building the CAP curve
write.csv(file = "G:\\Data Science A-Z\\Modelling\\predicted.csv",x = data.frame(my_df[c(1:10000),]$Exited,predicted_train))

#Confusion Matrix
predict_abs <- ifelse(predicted_train > 0.5,1,0)
table(my_df[c(1:10000),]$Exited,predict_abs)

#Use test data to assess the model.
#Here we simple select the last 1000 records as our test data.

test_df <- my_df[c(10001:11000),]
head(test_df)

#Predict on the test data set to build a CAP curve and see accuracy
predicted_test <- predict(model6, newdata = test_df,type='response')
head(predicted_test)

#Write the actual exited vs predicted probablities for the test dataset to a csv for building the CAP curve
write.csv(file = "G:\\Data Science A-Z\\Modelling\\test_predicted.csv",x = data.frame(my_df[c(10001:11000),]$Exited,predicted_test))

#Confusion Matrix
predict_test_abs <- ifelse(predicted_test > 0.5,1,0)
table(my_df[c(10001:11000),]$Exited,predict_test_abs)

#We can see a slight drop in performance but even then our model still is a good model.
#We need to keep in mind the way we select our test data. Here, we simply selected the last 1000 records, however in real world, 
#we apply other methods to select better test data from our sample. Usually a k-fold cross validation works best.

#CAP Curve is especially helpful in comparing and choosing between multiple models.
#Area under the CAP curve i.e the curve which increases at a faster rate is better than others.
#CAP curve helps in avoiding the accuracy paradox associated with solely relying on accuracy from confusion matrix.

#ROCR curve also helps in assesing models. The Area under curve(AUC) is the parameter which is used to compare multiple models.
