args<-commandArgs(TRUE);task<-normalizePath(args[1]);root<-normalizePath(args[2]);rev<-dirname(root)
.libPaths(strsplit(Sys.getenv("FRONTIER_R_LIBS"),.Platform$path.sep,fixed=TRUE)[[1]])
suppressPackageStartupMessages(library(sfaR));suppressPackageStartupMessages(library(jsonlite));options(digits=17)
dat<-read.csv(file.path(task,'train.csv'));x<-as.matrix(dat[grep('^x[0-9]+$',names(dat))]);y<-log(dat$y);lx<-log(x);n<-nrow(x);m<-ncol(x);result<-list()
folder<-file.path(task,'sfaR_baseline');dir.create(folder,showWarnings=FALSE)
for(form in c('CD','TL')){
  a<-lx
  if(form=='TL'){cr<-combn(m,2);a<-cbind(lx,.5*lx^2,sapply(seq_len(ncol(cr)),function(k)lx[,cr[1,k]]*lx[,cr[2,k]]))}
  X<-cbind(1,a);ols<-lm.fit(X,y);sd<-sqrt(mean(ols$residuals^2));threshold<-128*.Machine$double.eps*max(dim(X))*max(1,max(abs(y)))
  for(dist in c('hnormal','exponential')){
    id<-paste('SFA',form,if(dist=='hnormal')'HN'else'EXP',sep='_');started<-proc.time()[3];warnings<-character()
    ans<-tryCatch(withCallingHandlers({
      if(ols$rank<ncol(X)||sd<=threshold){list(status='DEGENERATE_OR_RANK_DEFICIENT',rank=ols$rank,ols_residual_sd=sd)}else{
        d<-data.frame(log_y=y,a);names(d)<-c('log_y',paste0('z',seq_len(ncol(a))));fm<-reformulate(names(d)[-1],'log_y')
        fit<-sfacross(fm,data=d,udist=dist,method='bfgs',itermax=2000,tol=1e-12)
        saveRDS(fit,file.path(folder,paste0(id,'.rds')));p<-as.numeric(coef(fit));writeBin(p,file.path(folder,paste0(id,'_params.f64')),size=8,endian='little')
        sc<-efficiencies(fit);saveRDS(sc,file.path(folder,paste0(id,'_scores.rds')));writeBin(as.double(cbind(sc$teBC,sc$teJLMS)),file.path(folder,paste0(id,'_scores.f64')),size=8,endian='little')
        list(status='COMPUTED_RAW_NOT_AUTO_QUALIFIED',optStatus=fit$optStatus,loglik=fit$mlLoglik,gradientNorm=fit$gradientNorm,coefficients=p)
      }
    },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(status='FAILED_RAW_PACKAGE',error=conditionMessage(e)))
    ans$seconds<-unname(proc.time()[3]-started);ans$warnings<-warnings;result[[id]]<-ans
    write_json(result,file.path(folder,'STATUS.json'),pretty=TRUE,auto_unbox=TRUE,digits=NA,na='null')
    cat(id,ans$status,ans$seconds,'\n')
  }
}
writeLines(capture.output(sessionInfo()),file.path(folder,'R_sessionInfo.txt'))
