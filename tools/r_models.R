args<-commandArgs(TRUE);task<-normalizePath(args[1]);root<-normalizePath(args[2]);rev<-dirname(root)
.libPaths(strsplit(Sys.getenv("FRONTIER_R_LIBS"),.Platform$path.sep,fixed=TRUE)[[1]])
options(digits=17);suppressPackageStartupMessages(library(jsonlite))
tr<-read.csv(file.path(task,'train.csv'),check.names=FALSE);ev<-read.csv(file.path(task,'eval.csv'),check.names=FALSE)
cfg<-fromJSON(file.path(task,'TASK.json'));xn<-grep('^x[0-9]+$',names(tr),value=TRUE)
x<-as.matrix(tr[xn]);y<-tr$y;xe<-as.matrix(ev[xn]);xx<-rbind(x,xe);n<-nrow(x);m<-ncol(x)
pred<-data.frame(row=seq_len(nrow(xx)),is_train=seq_len(nrow(xx))<=n);status<-list()
flush_predictions<-function(){
 writeBin(as.double(as.matrix(pred)),file.path(task,'R_predictions.f64'),size=8,endian='little')
 write_json(list(rows=nrow(pred),columns=names(pred)),file.path(task,'R_predictions_shape.json'),auto_unbox=TRUE)
}
save_status<-function()write_json(status,file.path(task,'r_status.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
attempt<-function(id,fun){
 w<-character();t<-proc.time()[3]
 ans<-tryCatch(withCallingHandlers(fun(),warning=function(e){w<<-c(w,conditionMessage(e));invokeRestart('muffleWarning')}),error=function(e)e)
 if(inherits(ans,'error'))status[[id]]<<-list(status='FAILED',error=conditionMessage(ans),strict_qualified=FALSE)
 else status[[id]]<<-c(list(status='COMPUTED'),ans)
 status[[id]]$elapsed_seconds<<-unname(proc.time()[3]-t);status[[id]]$warnings<<-w;save_status();flush_predictions()
 cat(id,status[[id]]$status,status[[id]]$elapsed_seconds,'\n')
}
attempt('FDH',function(){
 raw<-as.numeric(Benchmarking::dea(xx,matrix(1,nrow(xx),1),RTS='fdh',ORIENTATION='out',XREF=x,YREF=matrix(y,ncol=1),FAST=TRUE))
 direct<-apply(xx,1,function(p){inside<-rowSums(sweep(x,2,p,'-')>0)==0;if(any(inside))max(y[inside])else NA_real_})
 mask<-is.finite(direct);z<-raw;z[!mask]<-NA_real_;err<-max(abs(z[mask]-direct[mask]))
 pred$FDH_frontier<<-z;pred$FDH_input_support<<-mask;pred$FDH_raw<<-raw
 saveRDS(list(package_raw=raw,direct=direct,mask=mask),file.path(task,'FDH_crosscheck.rds'))
 list(backend='Benchmarking0.33',direct_max_error=err,strict_qualified=all(is.finite(z[mask]))&&err<=1e-7,package_converged=NA)
})
trace_names<-c('generateLv','scores','SERules','selectTk')
for(nm in trace_names)trace(nm,where=asNamespace('eat'),print=FALSE,exit=substitute(saveRDS(returnValue(),DEST),list(DEST=file.path(task,paste0('eat_CV_',nm,'.rds')))))
fit<-NULL
attempt('EAT',function(){
 RNGkind('Mersenne-Twister','Inversion','Rejection');set.seed(as.integer(cfg$eat_seed));saveRDS(.Random.seed,file.path(task,'eat_RNG_before.rds'))
 fit_start<-proc.time()[3];fit<<-eat::EAT(tr,x=seq_len(m),y=m+1L,numStop=5,fold=5);fit_seconds<-unname(proc.time()[3]-fit_start)
 saveRDS(.Random.seed,file.path(task,'eat_RNG_after.rds'));saveRDS(fit,file.path(task,'EAT_shared_tree.rds'))
 pred_start<-proc.time()[3];z<-as.numeric(predict(fit,newdata=as.data.frame(xx),x=seq_len(m))$y_pred);predict_seconds<-unname(proc.time()[3]-pred_start)
 pred$EAT_frontier<<-z
 # Independent row traversal of the saved selected tree; no call to predict.
 direct<-vapply(seq_len(nrow(xx)),function(i){
   k<-1L
   while(fit$tree[[k]]$SL!=-1){node<-fit$tree[[k]];nextid<-if(xx[i,node$xi]<as.numeric(node$s))node$SL else node$SR
     k<-which(vapply(fit$tree,function(t)as.numeric(t$id),numeric(1))==nextid)
   };as.numeric(fit$tree[[k]]$y)
 },numeric(1))
 err<-max(abs(z-direct));envelop<-min(z[seq_len(n)]-y)
 write_json(fit$tree,file.path(task,'EAT_tree.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
 list(seed=cfg$eat_seed,backend='eat0.1.4',numStop=5,fold=5,direct_tree_max_error=err,fit_CV_seconds=fit_seconds,predict_seconds=predict_seconds,
      train_envelopment_min=envelop,strict_qualified=all(is.finite(z))&&err<=1e-10&&envelop>=-1e-7,package_converged=NA)
})
for(nm in trace_names)untrace(nm,where=asNamespace('eat'))
attempt('CEAT',function(){
 if(is.null(fit))stop('EAT tree unavailable')
 raw<-as.numeric(eat::efficiencyCEAT(data.frame(xx,y=1,check.names=FALSE),x=seq_len(m),y=m+1L,object=fit,scores_model='BCC.OUT',digits=12,DEA=FALSE,print.table=FALSE))
 z<-raw;z[!is.finite(z)|z<=-1e20]<-NA_real_
 ax<-as.matrix(fit$model$a);ay<-as.numeric(fit$model$y)
 write.table(data.frame(ax,y=ay),file.path(task,'CEAT_leaves.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
 leaves<-cbind(ax,y=ay);writeBin(as.double(leaves),file.path(task,'CEAT_leaves.f64'),size=8,endian='little')
 write_json(list(rows=nrow(leaves),columns=ncol(leaves)),file.path(task,'CEAT_leaves_shape.json'),auto_unbox=TRUE)
 pred$CEAT_frontier<<-z;pred$CEAT_raw<<-raw
 list(backend='eat0.1.4',shared_tree='EAT_shared_tree.rds',strict_qualified=FALSE,qualification_pending='all-query independent leaf-hull LP certificate',package_converged=NA)
})
saveRDS(pred,file.path(task,'R_predictions_raw.rds'))
writeBin(as.double(as.matrix(pred)),file.path(task,'R_predictions.f64'),size=8,endian='little')
write_json(list(rows=nrow(pred),columns=names(pred),storage='little_endian_binary64_column_major',authoritative='R_predictions_raw.rds and R_predictions.f64'),file.path(task,'R_predictions_shape.json'),auto_unbox=TRUE,pretty=TRUE)
write.table(pred,file.path(task,'r_predictions.tsv'),row.names=FALSE,sep='\t',quote=FALSE,na='NA')
writeLines(capture.output(sessionInfo()),file.path(task,'R_sessionInfo.txt'));save_status()
