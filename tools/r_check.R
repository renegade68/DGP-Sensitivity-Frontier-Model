args<-commandArgs(TRUE);task<-normalizePath(args[1]);root<-normalizePath(args[2]);rev<-dirname(root)
.libPaths(strsplit(Sys.getenv("FRONTIER_R_LIBS"),.Platform$path.sep,fixed=TRUE)[[1]])
suppressPackageStartupMessages(library(sfaR));suppressPackageStartupMessages(library(jsonlite))
tr<-read.csv(file.path(task,'train.csv'));ev<-read.csv(file.path(task,'eval.csv'));xn<-grep('^x[0-9]+$',names(tr),value=TRUE);x<-as.matrix(tr[xn]);xe<-as.matrix(ev[xn]);ly<-log(tr$y);n<-length(ly);m<-ncol(x);out<-list()
for(form in c('CD','TL')){
 a<-log(rbind(x,xe));A<-a
 if(form=='TL'){cr<-combn(m,2);A<-cbind(a,.5*a^2,sapply(seq_len(ncol(cr)),function(k)a[,cr[1,k]]*a[,cr[2,k]]))}
 X<-cbind(1,A);K<-ncol(X)
 for(dist in c('HN','EXP')){
  id<-paste('SFA',form,dist,sep='_');folder<-file.path(task,id);if(!file.exists(file.path(folder,'SELECTED.f64')))next;p<-readBin(file.path(folder,'SELECTED.f64'),what='double',n=K+2,size=8,endian='little');b<-p[seq_len(K)];su<-p[K+1];sv<-p[K+2]
  f<-exp(as.numeric(X%*%b));raw<-readBin(file.path(folder,'FRONTIER.f64'),what='double',n=nrow(X),size=8,endian='little');front_error<-max(abs(f-raw)/(1+abs(f)))
  meta<-fromJSON(file.path(folder,'STATUS.json'));e<-ly-as.numeric(X[seq_len(n),,drop=FALSE]%*%b);native<-NULL
  if(su==0&&sv==0){ll<-NULL;ll_error<-NULL;ll_pass<-TRUE;note<-'deterministic limit; no finite regular loglik'}else{
   if(su==0){ll<-sum(dnorm(e,sd=sv,log=TRUE));note<-'Gaussian boundary'}else if(sv==0){
    ll<-if(dist=='HN')sum(log(2)+dnorm(e,sd=su,log=TRUE))else sum(-log(su)+e/su);note<-'one-sided boundary density, primal support separately certified'
   }else{
    sig<-sqrt(su^2+sv^2)
    ll<-if(dist=='HN')sum(log(2)-log(sig)+dnorm(e/sig,log=TRUE)+pnorm(-su/sv*e/sig,log.p=TRUE))else sum(-log(su)+e/su+.5*(sv/su)^2+pnorm(-e/sv-sv/su,log.p=TRUE))
    fn<-get(if(dist=='HN')'chalfnormlike'else'cexponormlike',asNamespace('sfaR'));pp<-c(b,2*log(su),2*log(sv))
    native<-sum(fn(parm=pp,nXvar=K,nuZUvar=1,nvZVvar=1,uHvar=matrix(1,n,1),vHvar=matrix(1,n,1),Yvar=ly,Xvar=X[seq_len(n),,drop=FALSE],wHvar=rep(1,n),S=1));note<-'independent R log-CDF identity; native package also recorded'
   }
   ll_error<-abs(ll-meta$loglik);ll_pass<-is.finite(ll_error)&&ll_error<=1e-7*(1+abs(meta$loglik))
  }
  out[[id]]<-list(frontier_all_point_scaled_error=front_error,independent_loglik=ll,loglik_abs_error=ll_error,native_package_loglik=native,qualified=front_error<=1e-10&&ll_pass,note=note)
 }
}
write_json(out,file.path(task,'R_INDEPENDENT_CHECK.json'),pretty=TRUE,auto_unbox=TRUE,digits=NA,na='null')
