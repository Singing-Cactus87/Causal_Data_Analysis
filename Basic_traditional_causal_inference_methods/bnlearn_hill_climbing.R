install.packages("bnlearn")
install.packages("qgraph")
install.packages("psychTools")

library(bnlearn)
library(qgraph)
library(psychTools)

#Loading Data
dt <- bnlearn::asia

dt

#Yes/No -> 수치형 자료로 변환
length(dt)
for (i in c(1:8)){
  dt[,i] <- as.numeric(dt[,i])
}

for (i in c(1:8)){
  dt[,i] <- scale(dt[,i], center=min(dt[,i]), scale=max(dt[,i])-min(dt[,i]))
}


head(dt)

colnames(dt)


#DAG 상 blacklist 생성

dtsub <- dt[,]

Labels <- colnames(dtsub)

blk_l <- matrix(c("X","S"),ncol=2,byrow=T)
colnames(blk_l) <- c("from","to")


#부트스트랩 기반 DAG HC 탐색 진행

set.seed(123123)
boot <- boot.strength(dtsub, R=100, algorithm="hc", algorithm.args=list(blacklist=blk_l))

qgraph(boot, nodeNames=Labels, legend.cex=0.6, edge.labels=T, asize=4, edge.color="black",threshold=0.2,vsize=8,edge.label.cex=0.7)

set.seed(123123)
extraction <- hc(dtsub,blacklist=blk_l)
Labels <- colnames(dt)
qgraph(extraction, nodeNames=Labels, legend.cex=0.5, asize=4, edge.color="black",vsize=8)