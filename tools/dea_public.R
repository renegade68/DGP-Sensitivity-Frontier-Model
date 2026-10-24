# Same public Benchmarking VRS output LP, with observed solver codes and certificates.
instrumented_dea <- function(x,y,q,scaling=NULL) {
  calls <- vector("list",nrow(q)); fun <- Benchmarking::dea
  env <- new.env(parent=environment(fun))
  solver <- getS3method("solve","lpExtPtr",envir=asNamespace("lpSolveAPI"))
  env$solve <- function(a,...) {
    k <- get("k",envir=parent.frame()); code <- solver(a,...)
    calls[[k]] <<- c(calls[[k]],as.integer(code)); code
  }
  environment(fun) <- env
  warning_text <- character(); fit <- NULL
  args <- list(X=q,Y=matrix(1,nrow(q),1),RTS="vrs",ORIENTATION="out",XREF=x,YREF=matrix(y,ncol=1),FAST=FALSE,DUAL=TRUE)
  if(!is.null(scaling))args$CONTROL<-list(scaling=scaling)
  output <- capture.output(fit <- withCallingHandlers(do.call(fun,args),warning=function(w){warning_text<<-c(warning_text,conditionMessage(w));invokeRestart("muffleWarning")}))
  list(fit=fit,codes=calls,stdout=output,warnings=warning_text,scaling=scaling)
}
certificates <- function(attempt,x,y,q) {
  n<-nrow(x);m<-ncol(x);f<-attempt$fit
  w<-f$primal[,(ncol(f$primal)-n+1):ncol(f$primal),drop=FALSE]
  beta<--f$dual[,2:(m+1),drop=FALSE]
  alpha<--f$dual[,m+3]+f$dual[,m+4]
  lower<-as.numeric(w%*%y);upper<-alpha+rowSums(beta*q)
  rowmax<-function(z) apply(z,1,max)
  pres<-pmax(rowmax(w%*%x-q),abs(rowSums(w)-1),rowmax(-w),0)
  dres<-pmax(rowmax(matrix(y,nrow(q),n,byrow=TRUE)-matrix(alpha,nrow(q),n)-beta%*%t(x)),rowmax(-beta),0)
  gap<-abs(upper-lower);value<-as.numeric(f$objval)
  finalcodes<-vapply(attempt$codes,function(v)if(length(v))tail(v,1)else -999L,integer(1))
  good<-finalcodes==0L & is.finite(value) & value>0 & is.finite(pres) & is.finite(dres) & is.finite(gap) & pres<=1e-8 & dres<=1e-8 & gap<=1e-7 & abs(value-lower)<=1e-7 & abs(value-upper)<=1e-7
  good[is.na(good)]<-FALSE
  list(value=value,weights=w,beta=beta,alpha=alpha,lower=lower,upper=upper,primal_residual=pres,dual_residual=dres,gap=gap,good=good,final_codes=finalcodes)
}
