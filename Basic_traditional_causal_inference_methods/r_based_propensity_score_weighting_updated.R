library(MatchIt)
library(WeightIt)
library(cobalt)
library(marginaleffects)

#PSM
dt1 <- MatchIt::lalonde
head(dt1,5)

psm1 <- matchit(treat~age+educ+nodegree+married+re74+factor(race),data=dt1,distance="glm",link="probit",replace=T,ratio=1)
bal.tab(psm1,m.threshold=0.1,v.threshold=2,un=T)

bal.plot(psm1, var.name="educ",which='both',grid=T)
bal.plot(psm1, var.name="married",which='both',grid=T)

love.plot(bal.tab(psm1,m.threshold=0.1,v.threshold=2,un=T),grid=T,abs=T,stars="raw")


###PSW

#단순 weighted regression 통한 PSW
#앞선 확보된 psm1으로부터  
paired = match.data(psm1)
PSW_reg = lm(re78~treat+age+educ+nodegree+married+re74+factor(race),data=paired,weights=paired$weights)
summary(PSW_reg)


# 수동으로 원하는 weighting 방식 선택하여 IPTW 기반 PSW 실행하기
ps_score = psm1$distance
ps_score = pmax(pmin(ps_score,1-1e-8),1e-8) #inf 방지용
IPTW_manual = ifelse(dt1$treat==1,1/ps_score,1/(1-ps_score))

#IPTW 기반 PSW 진행
PSW_reg = lm(re78~treat+age+educ+nodegree+married+re74+factor(race),data=dt1,weights=IPTW_manual)
summary(PSW_reg)

#IPTW 기반 PSW에 따른 covariance balance 확인

bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=IPTW_manual,data=dt1,s.d.denom="pooled")
love.plot(bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=IPTW_manual,data=dt1,s.d.denom="pooled",m.threshold=0.1,v.threshold=2),grid=T,abs=F,stars="raw")
love.plot(bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=IPTW_manual,data=dt1,s.d.denom="pooled",m.threshold=0.1,v.threshold=2),grid=T,abs=T,stars="raw")

# WeightIt 패키지로 원하는 weighting 방식 선택하여 PSW 실행하기 (ATT 방식)
PSW_w <- weightit(treat ~ age+educ+nodegree+married+re74+factor(race), data = dt1, method = "ps", estimand = "ATT")
PSW_reg2 = lm(re78~treat+age+educ+nodegree+married+re74+factor(race),data=dt1,weights=PSW_w$weights)
#PSW_reg2 = lm_weightit(re78~treat+age+educ+nodegree+married+re74+factor(race),data=dt1,weightit=PSW_w)
summary(PSW_reg2)

bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=PSW_w$weights,data=dt1,s.d.denom="pooled")

love.plot(bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=PSW_w$weights,data=dt1,s.d.denom="pooled",m.threshold=0.1,v.threshold=2),grid=T,abs=F,stars="raw")
love.plot(bal.tab(treat~age+educ+nodegree+married+re74+factor(race), weights=PSW_w$weights,data=dt1,s.d.denom="pooled",m.threshold=0.1,v.threshold=2),grid=T,abs=T,stars="raw")


#조금 더 명확한 검정일 필요할 경우 lmtest 패키지 내에 있는 coeftest 사용. (vcovHC는 sandwich 패키지 내 존재)
library(lmtest)
library(sandwich)
coeftest(PSW_reg2,vcov. = vcovHC(PSW_reg2,type="HC1")) #robust Standard Error 반환, t-test 진행
#여기서 HC1은 자유도 조정이 들어간 Heteroskedastic consistent estimator 말함.
#Robust SE란, 모형에서 heteroskedasticity를 고려한 회귀 진행 시의 SE 의미: 더욱 실제 데이터 분석 환경에 맞는 분석 결과 반환 가능