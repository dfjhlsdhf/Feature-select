library(copent) # Copula Entropy
library(energy) # Distance Correlation
library(dHSIC) # Hilbert-Schmidt Independence Criterion
## for additional tests
library(HHG) # Heller-Heller-Gorfine Tests of Independence
library(independence) # Hoeffding's D test or Bergsma-Dassios T* sign covariance
library(Ball) # Ball correlation
#把四个excel的fea都拼成一个

# heart1 = as.matrix( rbind(h1,h2,h3,h4) )
heart1=read.xlsx("delta_fea.xlsx",sheetIndex = 1)
m = dim(heart1)[1]#行数
n = dim(heart1)[2]#列数

#statistical dependence with attr #58
#Copula Entropy
l = 50
ce58 = rep(0,n)
for (i in 1:n){
  for (j in 1:l){
    data2 = heart1[,c(i,69)]
    data2[,1] = data2[,1] + max(abs(data2[,1])) * 0.000001 * rnorm(m)
    data2[,2] = data2[,2] + max(abs(data2[,2])) * 0.000001 * rnorm(m)
    ce58[i] = ce58[i] + copent(data2)
  }
}
ce58 = ce58 / l
ce58[c(1,2,69)] = min(ce58)
print(ce58)

# ce
x11(width = 10, height = 5)
plot(ce58, xlab = "Variable", ylab = "Copula Entropy", xaxt = 'n')
lines(ce58)
axis(side = 1, at = c(seq(1,69, by = 1)), labels = c(seq(1,69, by = 1)))
th16a = rep(ce58[37],69)
lines(th16a, col = "red")
