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

love.plot(bal.tab(psm1,m.threshold=0.1,v.threshold=2,un=T),grid=T,abs=T,stars="raw")


###PSW

#단순 weighted regression 통한 PSW
#앞선 확보된 psm1으로부터  
paired = match.data(psm1)
PSW_reg = lm(re78~treat+age+educ+nodegree+married+re74+factor(race),data=paired,weights=paired$weights)
summary(PSW_reg)


# cobalt 패키지 활용하여 원하는 weighting 방식 선택하여 PSW 실행하기

IPTW_p = get.w(psm1,estimand="ATE")
length(IPTW_p) #데이터 length 확인
length(dt1$treat) #데이터 length 확인

#PSW 진행
PSW_reg = lm(re78~treat+age+educ+nodegree+married+re74+factor(race),data=dt1,weights=IPTW_p)
summary(PSW_reg)

#PSW에 따른 covariance balance 확인

bal.tab(psm1, weights=IPTW_p)

love.plot(bal.tab(psm1, weights=IPTW_p,m.threshold=0.1,v.threshold=2),grid=T,abs=F,stars="raw")
love.plot(bal.tab(psm1, weights=IPTW_p,m.threshold=0.1,v.threshold=2),grid=T,abs=T,stars="raw")

#조금 더 명확한 검정일 필요할 경우 lmtest 패키지 내에 있는 coeftest 사용. (vcovHC는 sandwich 패키지 내 존재)
library(lmtest)
library(sandwich)
coeftest(PSW_reg,vcov. = vcovHC(PSW_reg,type="HC1")) #robust Standard Error 반환, t-test 진행
#여기서 HC1은 자유도 조정이 들어간 Heteroskedastic consistent estimator 말함.
#Robust SE란, 모형에서 heteroskedasticity를 고려한 회귀 진행 시의 SE 의미: 더욱 실제 데이터 분석 환경에 맞는 분석 결과 반환 가능