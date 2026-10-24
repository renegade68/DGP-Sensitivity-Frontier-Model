args<-commandArgs(TRUE);out<-normalizePath(args[1]);rev<-dirname(out)
.libPaths(c(file.path(rev,'efficiency_benchmark_20261007/environment/R-library'),.libPaths()))
suppressPackageStartupMessages(library(sfaR));suppressPackageStartupMessages(library(frontier));suppressPackageStartupMessages(library(jsonlite))
options(digits=17);inp<-read.csv(file.path(out,'INPUTS.csv'));results<-list()
writeLines(capture.output(sessionInfo()),file.path(out,'R_SESSION.txt'))
for(ii in seq_len(nrow(inp))){
 d0<-read.csv(file.path(inp$directory[ii],'train.csv'));x<-as.matrix(d0[grep('^x[0-9]+$',names(d0))]);y<-log(d0$y);lx<-log(x);m<-ncol(x)
 for(form in c('CD','TL')){
  a<-lx
  if(form=='TL'){cr<-combn(m,2);a<-cbind(lx,.5*lx^2,sapply(seq_len(ncol(cr)),function(k)lx[,cr[1,k]]*lx[,cr[2,k]]))}
  X<-cbind(1,a);K<-ncol(X);ols<-lm.fit(X,y);sd<-sqrt(mean(ols$residuals^2));d<-data.frame(log_y=y,a);names(d)<-c('log_y',paste0('z',seq_len(ncol(a))));fm<-reformulate(names(d)[-1],'log_y')
  for(dist in c('HN','EXP')){
   id<-paste('SFA',form,dist,sep='_')
   for(engine in c('sfaR_nlminb',if(dist=='HN')'frontier')){
    starts<-if(engine=='frontier')c(NA,1,10)else c(.1,1,10)
    for(lam in starts){
     warnings<-character();started<-proc.time()[3]
     result<-tryCatch(withCallingHandlers({
      initial<-NULL
      if(!is.na(lam)){
       cmean<-if(dist=='HN')sqrt(2/pi)else 1;cvar<-if(dist=='HN')1-2/pi else 1
       sv<-sd/sqrt(1+cvar*lam^2);su<-sv*lam;b<-ols$coefficients;b[1]<-b[1]+cmean*su
       initial<-if(engine=='frontier')c(b,su^2+sv^2,su^2/(su^2+sv^2))else c(b,2*log(su),2*log(sv))
      }
      if(engine=='frontier'){
       fit<-frontier::sfa(fm,data=d,startVal=initial,maxit=2000,tol=1e-10,searchTol=1e-6)
       pp<-as.numeric(fit$mleParam);s2<-pp[K+1];gamma<-pp[K+2]
       list(status='returned',reported_loglik=fit$mleLogl,beta=pp[1:K],sigma_u=sqrt(s2*gamma),sigma_v=sqrt(s2*(1-gamma)),code=fit$code,iterations=fit$nIter)
      }else{
       fit<-sfaR::sfacross(fm,data=d,udist=if(dist=='HN')'hnormal'else'exponential',method='nlminb',start=initial,itermax=2000,tol=1e-12,gradtol=1e-8)
       pp<-as.numeric(fit$mlParam)
       list(status='returned',reported_loglik=fit$mlLoglik,beta=pp[1:K],sigma_u=exp(pp[K+1]/2),sigma_v=exp(pp[K+2]/2),code=fit$optStatus,iterations=fit$nIter,gradientNorm=fit$gradientNorm)
      }
     },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(status='error',error=conditionMessage(e)))
     result$task<-inp$task[ii];result$model<-id;result$engine<-engine;result$start<-if(is.na(lam))'package_default'else paste0('OLS_lambda_',lam);result$seconds<-unname(proc.time()[3]-started);result$warnings<-warnings
     results[[length(results)+1L]]<-result
     write_json(results,file.path(out,'R_RESULTS.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null',null='null')
     cat(length(results),inp$task[ii],id,engine,result$start,result$status,result$seconds,'\n');flush.console()
    }
   }
  }
 }
}
saveRDS(results,file.path(out,'R_RESULTS.rds'))
write_json(list(fits=length(results),complete=TRUE),file.path(out,'R_DONE.json'),auto_unbox=TRUE,pretty=TRUE)

