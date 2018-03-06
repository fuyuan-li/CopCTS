# # rcopula.Clayton = function(t,u,theta)
# # {
# #   v = ((t^(-theta/(theta+1))-1)*u^(-theta)+1)^(-1/theta)
# #   return(v)
# # }
# # Judy.GenDat.Clayton.orig <- function(theta,df,N){
# #   Y <- NULL
# #   v <- runif(1)
# #   for(i in 1:N){
# #     Y <- c(Y, qt(v,df))
# #     u = max(v, 0.000001)
# #     t = runif(1)
# #     v = rcopula.Clayton(t,u,theta)
# #   }
# #   return(Y)
# # }
# # # rcopula.Frank = function(t,u,theta){
# # #   #t-th quantile of V given U=u
# # #   # a = pmax(theta,0.000001)
# # #   a = theta
# # #   A = (1-exp(-a))/((1/t-1)*exp(-a*u)+1)
# # #   v = -1/a*log(1-A)
# # #   return(v)
# # # }
# # #
# # # GenDat.Frank.orig <- function(theta,df,N){
# # #   a = theta
# # #
# # #   Y <- NULL
# # #   v <- runif(1)
# # #   for(i in 1:N){
# # #     Y <- c(Y, qt(v,df))
# # #     u = max(v, 0.000001)
# # #     t = runif(1)
# # #     # v = rcopula.Frank(t,u,a)
# # #     v = myrCopula.frankCopula(t,u,a)
# # #   }
# # #   return(Y)
# # # }
# # # set.seed(200)
# # # dat <- Judy.GenDat.Clayton.orig(1,df = 3,N = 100)
# # # # dat <- GenDat.Frank.orig(theta = 500,df = 3,N = 100)
# # # d = -0.7
# # # Y <- pmax(dat,d)
# # # delta <- (Y<=d)
# # #
# # #
# # #
# # # library(copula)
# # # library(msm)
# # # cop = claytonCopula()
# # # getCopC(cop,interval = c(0,400),Yc = Y,d,delta,nIS = 5)
# # # optimize(f = CopLikelihood.claytonCopula,interval = c(0,400),copula=cop
# # #          ,Yc=Y,d=d,delta=delta,nIS=500)
# # #
# # # ints <- cbind(c(0,400),c(1,11),c(2,200))
# # #
# # # estCopC(cop = c("Clayton"),interval =c(-1,200),Yc = Y,d,delta,nIS = 500)
# # # estCopC(cop = c("Gaussian"),interval =c(-1,1),Yc = Y,d,delta,nIS = 5)
# # # estCopC(cop = c("Gumbel"),interval =c(1,200),Yc = Y,d,delta,nIS = 5)
# # # estCopC(cop = c("Joe"),interval =c(1,Inf),Yc = Y,d,delta,nIS = 5)
# # # estCopC(cop = c("Frank"),interval =c(1,200),Yc = Y,d,delta,nIS = 500)
# # # estCopC(cop = c("Frank"),interval =c(-Inf,Inf),Yc = Y,d,delta,nIS = 500)
# # # estCopC(cop = c("Clayton"),Yc = Y,d,delta,interval = c(0,2),MARGIN = pt,df=3)
# # # estCopC(cop = c("Clayton"),Yc = Y,d,delta,interval = c(0,2))
# # # estCopC(cop = c("Clayton"),Yc = Y,d,delta)
# # #
# # # optimize(f = CopLikelihood.frankCopula,interval = c(1,2000),copula=frankCopula(),Yc=Y,d=d,delta=delta,nIS=5)
# # # CopLikelihood.frankCopula(frankCopula(),2000,Y,d,delta,nIS=5)
# # # CopLikelihood.frankCopula(frankCopula(),1000,Y,d,delta,nIS=5)
# # # CopLikelihood.frankCopula(frankCopula(),500,Y,d,delta,nIS=5)
# # # CopLikelihood.frankCopula(frankCopula(),1.84,Y,d,delta,nIS=5)
# # # Yc = Y
# # # N = length(Yc)
# # # if(is.null(jumps)){jumps <- c(TRUE,rep(FALSE,N-1))}
# # # Info <- CopCens(delta,jumps)
# # # nUC = sum(Info[,4]==0)
# # # nCen = sum(Info[,4]==1 | Info[,4]==2)
# # # Info.UC = Info[Info[,4]==0,]
# # # Info.C = Info[Info[,4]!=0,]
# # # if(nUC==1) Info.UC=matrix(Info.UC, ncol=4)
# # # if(nCen==1) Info.C=matrix(Info.C, ncol=4)
# # # if(is.null(MARGIN)) MARGIN=function(x){ecdf(Yc)(x)*N/(N+1)}
# # # FF <- MARGIN(Yc)
# # # Pi = MARGIN(d)
# # # logL1 = 0
# # # check = NULL
# # # for(j in 1:nUC){
# # #   idx = Info.UC[j,1]:Info.UC[j,2]
# # #   if(length(idx)>1){
# # #     for(k in 2:length(idx)){
# # #       # check <- c(check,pmax(1e-7,mydCopula.frankCopula(FF[idx[k-1]],FF[idx[k]],theta)))
# # #       logL1 <- logL1 + log(mydCopula.frankCopula(FF[idx[k-1]],FF[idx[k]],theta))
# # #     }
# # #   }
# # # }
# # # logL2 = 0
# # # if(nCen >= 1){
# # #   for(j in 1:nCen){
# # #     idx = Info.C[j,1]:Info.C[j,2]
# # #     nidx = length(idx)
# # #     if(1==Info.C[j,4]){print(j)
# # #       a = FF[idx[1]-1];b = FF[idx[nidx]+1]
# # #       INT = CopInt.frankCopula(start=a,upper=rep(Pi,nidx),theta,nIS,end=b)
# # #     }
# # #     else if(2==Info.C[j,4]){
# # #       a = FF[idx[1]-1]
# # #       INT = CopInt.frankCopula(start=a,upper=rep(Pi,nidx),theta,nIS)
# # #     }
# # #     logL2 <- logL2 + log(INT)
# # #   }
# # # }
# # # logL2
# # # #####
# # # CopInt.frankCopula(start=0.6,upper=rep(0.3,2),theta =90,nIS=5,end=0.8)
# # # m = length(upper)
# # # v2 = rep(start,nIS)
# # # h = rep(1,nIS)
# # # for(i in 1:m)
# # # {
# # #   v1 = v2
# # #   v2=rtcCopula.frankCopula(v1, theta, upper[i])
# # #   h = h*mycCopula.frankCopula(upper[i],v1,theta)
# # # }
# # # if(!is.na(end)){
# # #   h = h* mydCopula.frankCopula(v2,end,theta)
# # # }
# # # mean(h)
# # #
# # # rtcCopula.frankCopula(u_vec = c(0.6,0.6), theta = 9000, Pi=0.3)
# # # n = length(u_vec)
# # # a = theta
# # # w = runif(n, 0, 1)
# # # mycCopula.frankCopula(Pi, u_vec, theta=2)
# # # u=u_vec
# # # pc = exp(-a*u)*(1-exp(-a*v))/(exp(-a*u)+exp(-a*v)-exp(-a)-exp(-a*(u+v)))
# # # w2 = w* mycCopula.frankCopula(Pi, u_vec, theta)
# # # A = (1-exp(-a))/((1/w2-1)*exp(-a*u_vec)+1)
# # # A = pmin(1-1e-6,A)
# # # v = -1/a*log(1-A)
# # #
# # #
# # # mycCopula.frankCopula(v = 0.6,u = 0.6,theta = 9000)
# # # exp(-a*u)*(1-exp(-a*v))/(exp(-a*u)+exp(-a*v)-exp(-a)-exp(-a*(u+v)))
# # # fz = (exp(-a*v)-exp(-a))/(exp(-a*u)*(1-exp(-a*v)))
# # # 1/(1+fz)
# # # xx = exp(-1245)
# # # ys <- fs <- NULL
# # # for(aa in seq(400,9000,length.out = 500)){
# # #   a=aa
# # #   fz = (exp(-a*v)-exp(-a))/(exp(-a*u)*(1-exp(-a*v)))
# # #   fs <- c(fs,fz)
# # #   yy = 1/(1+fz)
# # #   ys <- c(ys,yy)
# # # }
# # # xs = seq(400,1000,length.out = 500)
# # # plot(x = xs[1:287],y = ys[1:287])
# # # #####
# # #
# # # logL2 = 0
# # # if(nCen >= 1){
# # #   for(j in 1:nCen){
# # #     idx = Info.C[j,1]:Info.C[j,2]
# # #     nidx = length(idx)
# # #     if(1==Info.C[j,4]){
# # #       a = FF[idx[1]-1];b = FF[idx[nidx]+1]
# # #       INT = max(0.00001, CopInt.frankCopula(start=a,upper=rep(Pi,nidx),theta,nIS,end=b))
# # #     }
# # #     else if(2==Info.C[j,4]){
# # #       a = FF[idx[1]-1]
# # #       INT = max(0.00001, CopInt.frankCopula(start=a,upper=rep(Pi,nidx),theta,nIS))
# # #     }
# # #     logL2 <- logL2 + log(INT)
# # #   }
# # # }
# # #
# # #
# # #
# # # CopLikelihood(claytonCopula(),theta = 400,Y,d,delta,nIS = 5)
# # #
# # #
# # #
# # #
# # #
# # #
# # #
# # # rtcCopula.gumbelCopula(u_vec = 0.99,a = 170,Pi = 0.9)
# # #
# # # n=length(u_vec)
# # # u = u_vec
# # # w = runif(n, 0, 1)
# # # t = w*mycCopula.gumbelCopula(rep(Pi,n),u_vec,a)
# # # len = n
# # # v = u
# # # eps = rep(1,n)
# # # nloop = 0
# # # while(sum(eps>0.000001) & (sum(eps>0.000001)>len/100 | max(eps) > 0.001) & nloop<10){
# # #   f = mycCopula.gumbelCopula(v,u,a)-t
# # #   f1 = mydCopula.gumbelCopula(u,v,a)
# # #   eps = abs(f)
# # #
# # #   v = v - f/f1
# # #   v=pmax(0,v) # to solve rcopula.gum.new(0.000001,0.000001,1.5) = -2.35329e-05
# # #   v=pmin(1,v) # to solve rcopula.gum.new(1-0.000001,1-0.000001,1.5) = 1.00002
# # #   nloop <- nloop+1
# # # }
# # # return(v)
# # # mydCopula.gumbelCopula(u = 0.6,v = 0.6,a = 600)
# # # U <- as.matrix(cbind(u,v))
# # # U = pmax(U,0.000001)
# # # U = pmin(U,0.999999)
# # # u1 = U[,1]
# # # u2 = U[,2]
# # # alpha=a
# # # exp(-((-log(u1))^alpha + (-log(u2))^alpha)^(1/alpha)) *
# # #   (((-log(u1))^alpha + (-log(u2))^alpha)^((1/alpha) - 1) *
# # #      ((1/alpha) * ((-log(u2))^(alpha - 1) * (alpha * (1/u2))))) *
# # #   (((-log(u1))^alpha + (-log(u2))^alpha)^((1/alpha) - 1) *
# # #      ((1/alpha) * ((-log(u1))^(alpha - 1) * (alpha * (1/u1))))) -
# # #   exp(-((-log(u1))^alpha + (-log(u2))^alpha)^(1/alpha)) *
# # #   (((-log(u1))^alpha + (-log(u2))^alpha)^(((1/alpha) -
# # #                                              1) - 1) * (((1/alpha) - 1) * ((-log(u2))^(alpha -
# # #                                                                                          1) * (alpha * (1/u2)))) * ((1/alpha) * ((-log(u1))^(alpha -
# # #                                                                                                                                                1) * (alpha * (1/u1)))))
# # #
# # # library(copula)
# # # dCopula(cbind(0.6,0.6),gumbelCopula(600))
# # # mycCopula.gumbelCopula(0.99,0.99,170)
# # # u<-v<-0.99
# # # a=170
# # # U = cbind(v,u)
# # # U <- as.matrix(U)
# # # U[,2] = pmax(U[,2],0.000001)
# # # U[,2] = pmin(U[,2],0.999999)
# # #
# # # A = -mypCopula.gumbelCopula(U,a)
# # # B = 1/a * (rowSums((-log(U))^a))^(1/a-1)
# # # C = a * (-log(U[,2]))^(a-1)
# # # return(A*B*C*(-1/U[,2]))
# #
# #
# #
# #
# # # set.seed(20)
# # # Y <- genLatentY(cop="Clayton",1,500,MARGIN.inv = qt,df=3)
# # # d <- -1
# # # Yc <- pmax(d,Y)
# # # delta <- (Y<=d)
# # # estCopC(cop = "Clayton",Yc,d,delta,nIS = 1000,interval = c(1,10))
# #
# #
# # Yc = TNH3
# # d = 0.02
# # delta = Delta
# # set.seed(1)
# # estCopC(cop="Joe",Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,interval = c(1,200),nIS=5)
# # CopLikelihood.joeCopula(copula=joeCopula(),theta = 199,Yc = TNH3,d = 0.02,delta = Delta,jumps = Indep,MARGIN = NULL,nIS=5)
# # # if(!is.na(copula@parameters)) theta=copula@parameters
# # N = length(Yc)
# # if(is.null(jumps)){jumps <- c(TRUE,rep(FALSE,N-1))}
# # Info <- CopCens(delta,jumps)
# # nUC = sum(Info[,4]==0)
# # nCen = sum(Info[,4]==1 | Info[,4]==2)
# # Info.UC = Info[Info[,4]==0,]
# # Info.C = Info[Info[,4]!=0,]
# # if(nUC==1) Info.UC=matrix(Info.UC, ncol=4)
# # if(nCen==1) Info.C=matrix(Info.C, ncol=4)
# # if(is.null(MARGIN)) MARGIN=function(x){ecdf(Yc)(x)*N/(N+1)}
# # FF <- MARGIN(Yc)
# # Pi = MARGIN(d)
# # logL1 = 0
# # for(j in 1:nUC){
# #   idx = Info.UC[j,1]:Info.UC[j,2]
# #   if(length(idx)>1){
# #     for(k in 2:length(idx)){
# #       # logL1 <- logL1 + log(mydCopula.joeCopula(FF[idx[k-1]],FF[idx[k]],theta))
# #       logL1 <- logL1 + log(pmax(1e-6,mydCopula.joeCopula(FF[idx[k-1]],FF[idx[k]],theta)))
# #     }
# #   }
# # }
# # U <- as.matrix(cbind(u,v))
# # U = pmax(U,0.000001)
# # U = pmin(U,0.999999)
# # u = U[,1]
# # v = U[,2]
# # # dC <- ((1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)^(1/a-2)
# # # dC <- dC * (1-u)^(a-1) * (1-v)^(a-1)
# # # dC <- dC * (a - 1 + (1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)
# # ua = (1-u)^a
# # va = (1-v)^a
# # dC = (ua+va-ua*va)^(1/a-2) *ua/(1-u)*va/(1-v) *(a - 1 + ua+va-ua*va)
# # dC[which(is.na(dC))] <- 1e-6
# #
# # logL2 = 0
# # if(nCen >= 1){
# #   for(j in 1:nCen){
# #     idx = Info.C[j,1]:Info.C[j,2]
# #     nidx = length(idx)
# #     if(1==Info.C[j,4]){
# #       a = FF[idx[1]-1];b = FF[idx[nidx]+1]
# #       INT = max(0.00001, CopInt.joeCopula(start=a,upper=rep(Pi,nidx),theta,nIS,end=b))
# #     }
# #     else if(2==Info.C[j,4]){
# #       a = FF[idx[1]-1]
# #       INT = max(0.00001, CopInt.joeCopula(start=a,upper=rep(Pi,nidx),theta,nIS))
# #     }
# #     logL2 <- logL2 + log(INT)
# #   }
# # }
# # logL = logL1+logL2
# #
# #
# #
# #
# # set.seed(1)
# # u_vec = rep(0.377,1)
# # Pi=0.3695
# # a=123
# # n = length(u_vec)
# # u = u_vec
# # w = runif(n, 0, 1)
# # t = w* (1-u)^(a-1) *(1-(1-Pi)^a) * (((1-u)^a+(1-Pi)^a-(1-u)^a*(1-Pi)^a)^(1/a-1))
# # v <- NULL
# # len = n
# # v = u
# # eps = rep(1,len)
# # nloop = 0
# # vv<-NULL
# # while(sum(eps>0.000001) & (sum(eps>0.000001)>len/100 | max(eps) > 0.001) & nloop<10){
# #   f = mycCopula.joeCopula(v,u,a)-t
# #   f1 = mydCopula.joeCopula(u,v,a)
# #   eps = abs(f)
# #   v = v - f/f1
# #   vv<-c(vv,v)
# #   # v[v<0] <- -v[v<0]
# #   v=pmax(0,v)
# #   eps[v<0] <- 1
# #   # v[v>1] <- 2-v[v>1]
# #   v=pmin(1,v)
# #   eps[v>1] <- 1
# #   nloop <- nloop+1
# # }
# #
# #
# # mycCopula.joeCopula(u=0.987,v=0.987,a = 182)
# # u = pmax(u,0.000001)
# # v = pmax(v,0.000001)
# # u = pmin(u,0.999999)
# # v = pmin(v,0.999999)
# # A = (1-u)^(a-1)
# # B = 1-(1-v)^a
# # C = pmax(1e-6,((1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a))
# # C = C^(1/a-1)
# # ret <- pmin(A*B*C,0.999999)
# # ret <- pmax(0.000001,ret)
# # # mydCopula.joeCopula
# # U <- as.matrix(cbind(u,v))
# # U = pmax(U,0.000001)
# # U = pmin(U,0.999999)
# # u = U[,1]
# # v = U[,2]
# # dC <- ((1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)^(1/a-2)*
# #   pmax(1e-6,(1-u)^(a-1)) * pmax(1e-6,(1-v)^(a-1))*
# #   (a - 1 + (1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)
# #
#
#
# library(copBasic)
# set.seed(20)
# Y <- genLatentY("Clayton",1,200,MARGIN.inv = qt,df=3)
# d = -1
# Yc = pmax(d,Y)
# delta = (Y<=d)
# selectCopC(cop.type = "Gumbel",Yc = Yc,d = d,delta = delta,nIS=5)
# intv.Clayton = c(0,20)
# intv.Gaussian = c(-1,1)
# intv.Gumbel = c(1,10)
# intv.Joe = c(1,10)
# intv.Frank = c(0,15)
# intervals = cbind(intv.Clayton,intv.Gaussian,intv.Gumbel,intv.Joe,intv.Frank)
# selectCopC(cop.type = c("Clayton","Gaussian","Frank"),Yc = TNH3,d = 0.02,delta = Delta,jumps = Indep,intervals = intervals[,c(1,2,5)],nIS=1)
# selectCopC(cop.type = c("Gumbel"),Yc = TNH3,d = 0.02,delta = Delta,jumps = Indep,intervals = intervals[,c(3)],nIS=1)
# estCopC(cop="Joe",Yc = TNH3,d = 0.02,delta = Delta,jumps = Indep,nIS = 1,interval = c(1,10))
#
# estCopC(cop="Gumbel",Yc,d,delta,nIS=5)
# getCopC(gumbelCopula(),interval=c(1,Inf),Yc,d,delta,nIS=5,jumps=NULL,MARGIN=NULL)
# getCopC.gumbelCopula(gumbelCopula(),interval=c(1,Inf),Yc,d,delta,nIS=5,jumps=NULL,MARGIN=NULL)
#
# interval=c(1,200)
# copula=gumbelCopula()
# optimize(f = CopLikelihood.gumbelCopula,interval = interval,copula=copula,Yc=Yc,d=d,delta=delta,nIS=5,jumps=NULL,MARGIN=NULL)
# CopLikelihood.gumbelCopula(copula,theta = 200,Yc,d,delta,nIS = 5,jumps = NULL,MARGIN = NULL)
# N = length(Yc)
# jumps=NULL
# if(is.null(jumps)){jumps <- c(TRUE,rep(FALSE,N-1))}
# Info <- CopCens(delta,jumps)
# nUC = sum(Info[,4]==0)
# nCen = sum(Info[,4]==1 | Info[,4]==2)
# Info.UC = Info[Info[,4]==0,]
# Info.C = Info[Info[,4]!=0,]
# if(nUC==1) Info.UC=matrix(Info.UC, ncol=4)
# if(nCen==1) Info.C=matrix(Info.C, ncol=4)
# if(is.null(MARGIN)) MARGIN=function(x){ecdf(Yc)(x)*N/(N+1)}
# FF <- MARGIN(Yc)
# Pi = MARGIN(d)
# logL1 = 0
# for(j in 1:nUC){
#   idx = Info.UC[j,1]:Info.UC[j,2]
#   if(length(idx)>1){
#     for(k in 2:length(idx)){
#       logL1 <- logL1 + log(mydCopula.gumbelCopula(FF[idx[k-1]],FF[idx[k]],theta))
#     }
#   }
# }
# logL2 = 0
# if(nCen >= 1){
#   for(j in 1:nCen){
#     idx = Info.C[j,1]:Info.C[j,2]
#     nidx = length(idx)
#     if(1==Info.C[j,4]){
#       a = FF[idx[1]-1];b = FF[idx[nidx]+1]
#       INT = max(0.00001, CopInt.gumbelCopula(start=a,upper=rep(Pi,nidx),theta,nIS,end=b))
#     }
#     else if(2==Info.C[j,4]){
#       a = FF[idx[1]-1]
#       INT = max(0.00001, CopInt.gumbelCopula(start=a,upper=rep(Pi,nidx),theta,nIS))
#     }
#     logL2 <- logL2 + log(INT)
#   }
# }
# #
#
#
#
#
# mydCopula.joeCopula = function(u,v,a){# for 2d vector, call mypCopula.Gumbel(cbind,a)
#   U <- as.matrix(cbind(u,v))
#   U = pmax(U,0.000001)
#   U = pmin(U,0.999999)
#
#   u = U[,1]
#   v = U[,2]
#   dC <- ((1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)^(1/a-2)
#   dC <- dC * (1-u)^(a-1) * (1-v)^(a-1)
#   dC <- dC * (a - 1 + (1-u)^a+(1-v)^a-(1-u)^a*(1-v)^a)
#   # dC = dCopula(U,joeCopula(a))
#   dC[which(is.na(dC))] <- 1e-6
#   return(dC)
# }
# rtcCopula.joeCopula = function(u_vec, a, Pi)
# {
#   n = length(u_vec)
#   u = u_vec
#   w = runif(n, 0, 1)
#   t = w* (1-u)^(a-1) *(1-(1-Pi)^a) * (((1-u)^a+(1-Pi)^a-(1-u)^a*(1-Pi)^a)^(1/a-1))
#   v <- NULL
#   len = n
#   v = u
#   eps = rep(1,len)
#   nloop = 0
#   while(sum(eps>0.000001) & (sum(eps>0.000001)>len/100 | max(eps) > 0.001) & nloop<10){
#     f = mycCopula.joeCopula(v,u,a)-t
#     f1 = mydCopula.joeCopula(u,v,a)
#     # f1 = dCopula(cbind(u,v),joeCopula(a))
#     eps = abs(f)
#     v = v - f/f1
#     # v[v<0] <- -v[v<0]
#     # eps[v<0] <- 1
#     # v[v>1] <- 2-v[v>1]
#     # eps[v>1] <- 1
#     v=pmin(1,v)
#     v=pmax(0,v)
#     nloop <- nloop+1
#   }
#   return(v)
# }
# t0 = proc.time()
# estCopC(cop="Joe",Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,interval = c(1,10),nIS=500)
# t1 = proc.time()
# t1-t0
#
# t0 = proc.time()
# estCopC(cop="Gumbel",Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,interval = c(1,10),nIS=1)
# t1 = proc.time()
# t1-t0

#
#
# attach(water)
# set.seed(1)
# estCopC(cop="Joe",Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,interval = c(1,10),nIS=500)
#
# t0 = proc.time()
# optimize(f = real.Gum,interval = c(1,10),Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,nIS=1)
# t1 = proc.time()
# t1-t0
#
# t0 = proc.time()
# optimize(f = CopLikelihood.joeCopula,interval = c(1,10),copula=joeCopula(),Yc=TNH3,d=0.02,delta=Delta,jumps=Indep,nIS=1,MARGIN=function(x){ecdf(Yc)(x)*N/(N+1)})
# t1 = proc.time()
# t1-t0
#
#
# set.seed(20)
# Y <- genLatentY(theta = 0.5,N = 100)
# d = -0.5
# delta = (Y<=d)
# # Yc = pmax(d,Y)
# selCop <- selectCopC(cop.type = c("Clayton","Gaussian"),Yc,d,delta,nIS=200)
# cq50 <- condQestCopC(0.6,Yc,d,delta,selCop$Selected)
# # f <- function(x,a,b,...){
# #   aa = a(x,...)
# #   bb = b(x,...)
# #   return(c(aa,bb))
# # }
#
# condQestCopC(0.5,Yc,d,delta,normalCopula(0.5))
# condQestCopC(0.5,Yc,d,delta,cop = "Gaussian",theta = 0.5)
# condQestCopC(0.5,Yc,d,delta,normalCopula(0.5),MARGIN = pnorm,MARGIN.inv = qnorm)




