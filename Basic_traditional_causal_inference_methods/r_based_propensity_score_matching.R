install.packages("MatchIt")
install.packages("cobalt")
install.packages("marginaleffects")

library(MatchIt)
library(cobalt)
library(marginaleffects)

#PSM
dt1 <- MatchIt::lalonde
head(dt1,5)

psm1 <- matchit(treat~age+educ+nodegree+married+re74+factor(race),data=dt1,distance="glm",link="probit",replace=T,ratio=1)
bal.tab(psm1,m.threshold=0.1,v.threshold=2,un=T)

bal.plot(psm1, var.name="educ",which='both',grid=T)
bal.plot(psm1, var.name="married",which='both',grid=T)

love.plot(bal.tab(psm1,m.threshold=0.1,,v.threshold=2,un=T),grid=T,abs=T,stars="raw")

dt2 <- match.data(psm1)

g1 <- dt2[dt2$treat==1,];g2 <- dt2[dt2$treat!=1,]

s_mean <-c(rep(0,1000))
set.seed(321)
for (i in 1:1000){
  s_mean[i] <- mean(sample(g1$re78,200,replace=T))-mean(sample(g2$re78,200,replace=T))
}
hist(s_mean,probability = T)
abline(v=mean(s_mean),col="red")
print(mean(s_mean))