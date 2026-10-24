# Exact original R CSV parsing, cached fixed evaluation panels; no estimator fits.
args<-commandArgs(TRUE);home<-normalizePath(args[1]);limit<-if(length(args)>1)as.integer(args[2])else Inf
suppressPackageStartupMessages(library(jsonlite));suppressPackageStartupMessages(library(digest))
rows<-read.csv(file.path(home,'INPUTS.csv'),check.names=FALSE);rows<-head(rows,min(nrow(rows),limit))
target<-file.path(home,'input_exports');outfile<-file.path(target,'R_INPUTS.bin');indexfile<-file.path(target,'R_INDEX.jsonl')
stopifnot(!file.exists(outfile),!file.exists(indexfile))
con<-file(outfile,'wb');idx<-file(indexfile,'wt');offset<-0;panels<-new.env();started<-proc.time()[3]
for(i in seq_len(nrow(rows))){
 r<-rows[i,];tr<-read.csv(file.path(r$directory,'train.csv'),check.names=FALSE);xn<-grep('^x[0-9]+$',names(tr),value=TRUE)
 x<-as.matrix(tr[xn]);y<-tr$y;stopifnot(nrow(x)==r$n,ncol(x)==r$m)
 if(!exists(r$eval_sha256,envir=panels,inherits=FALSE)){
  ev<-as.matrix(read.csv(file.path(r$directory,'eval.csv'),check.names=FALSE)[xn]);path<-file.path(target,paste0(r$eval_sha256,'.f64'))
  writeBin(as.double(ev),path,size=8,endian='little');assign(r$eval_sha256,nrow(ev),envir=panels)
 }
 saved<-readRDS(file.path(r$directory,'FDH_crosscheck.rds'));direct<-saved$direct;direct[!is.finite(direct)]<-0
 # Canonical unsupported entries are zero for the byte hash only, not predictions.
 dh<-digest(writeBin(as.double(direct),raw(),size=8,endian='little'),algo='sha256',serialize=FALSE)
 values<-c(as.double(x),as.double(y));writeBin(values,con,size=8,endian='little')
 rec<-list(task=r$task,offset=offset,n=r$n,m=r$m,eval_sha256=r$eval_sha256,eval_rows=get(r$eval_sha256,envir=panels),saved_direct_sha256=dh)
 writeLines(toJSON(rec,auto_unbox=TRUE,digits=NA),idx);offset<-offset+length(values)*8
 if(i%%500==0){flush(con);flush(idx);cat('R_EXPORT',i,'/',nrow(rows),'seconds',proc.time()[3]-started,'\n');flush.console()}
}
close(con);close(idx)
write_json(list(status='COMPLETE',datasets=nrow(rows),bytes=offset,evaluation_panels=length(ls(panels)),seconds=unname(proc.time()[3]-started)),file.path(target,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE)
