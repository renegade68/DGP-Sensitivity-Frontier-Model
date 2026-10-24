args<-commandArgs(TRUE);out<-normalizePath(args[1]);rev<-dirname(out);prior<-file.path(rev,'sfa_independent_validation_20261024')
.libPaths(c(file.path(rev,'efficiency_benchmark_20261007/environment/R-library'),.libPaths()))
suppressPackageStartupMessages(library(sfaR));suppressPackageStartupMessages(library(jsonlite))
selected<-read.csv(file.path(out,'SELECTED_R.csv'));raw<-readRDS(file.path(prior,'R_RESULTS.rds'));results<-list();values<-numeric();index<-list()
for(i in seq_len(nrow(selected))){
 row<-selected[i,];if(row$engine!='sfaR_nlminb')next
 z<-raw[[row$index0+1L]];warnings<-character()
 template_path<-file.path(row$directory,'sfaR_baseline',paste0(row$model,'.rds'))
 a<-readRDS(template_path);stopifnot(length(z$beta)==a$nXvar,is.finite(z$sigma_u),is.finite(z$sigma_v))
 # Reuse only formula/data metadata in memory; no fitting and no file overwrite.
 a$mlParam<-c(z$beta,2*log(z$sigma_u),2*log(z$sigma_v))
 ans<-tryCatch(withCallingHandlers({
   sc<-sfaR::efficiencies(a)
   mm<-model.matrix(a$formula,data=a$dataTable,rhs=1)
   fp<-exp(as.numeric(mm%*%z$beta))
   cbind(BC=sc$teBC,JLMS=sc$teJLMS,frontier=fp)
 },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)e)
 rec<-list(index0=row$index0,task=row$task,model=row$model,template=template_path,warnings=warnings)
 if(inherits(ans,'error')){rec$status<-'error';rec$error<-conditionMessage(ans)}else{
  rec$status<-'returned';rec$rows<-nrow(ans);rec$columns<-colnames(ans);rec$nonfinite_BC<-sum(!is.finite(ans[,1]));rec$nonfinite_JLMS<-sum(!is.finite(ans[,2]))
  index[[length(index)+1L]]<-list(index0=row$index0,offset=length(values),rows=nrow(ans),columns=3)
  values<-c(values,as.double(ans))
 }
 results[[length(results)+1L]]<-rec
}
writeBin(as.double(values),file.path(out,'NATIVE_SFAR_SCORES.f64'),size=8,endian='little')
write_json(index,file.path(out,'NATIVE_SFAR_INDEX.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
write_json(results,file.path(out,'NATIVE_SFAR_STATUS.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
writeLines(capture.output(sessionInfo()),file.path(out,'R_SESSION.txt'))
writeLines(deparse(get('chalfnormeff',asNamespace('sfaR'))),file.path(out,'native_HN_score_source.txt'))
writeLines(deparse(get('cexponormeff',asNamespace('sfaR'))),file.path(out,'native_EXP_score_source.txt'))
cat('Native score evaluations:',length(results),'\n')

