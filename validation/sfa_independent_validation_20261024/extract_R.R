args<-commandArgs(TRUE);out<-normalizePath(args[1]);rev<-dirname(out)
.libPaths(c(file.path(rev,'efficiency_benchmark_20261007/environment/R-library'),.libPaths()))
suppressPackageStartupMessages(library(jsonlite))
r<-readRDS(file.path(out,'R_RESULTS.rds'));values<-numeric();index<-list()
for(i in seq_along(r)){
 z<-r[[i]];v<-if(z$status=='returned')c(z$beta,z$sigma_u,z$sigma_v,z$reported_loglik)else numeric()
 index[[i]]<-list(index0=i-1L,offset=length(values),length=length(v));values<-c(values,v)
}
writeBin(as.double(values),file.path(out,'R_PARAMS.f64'),size=8,endian='little')
write_json(index,file.path(out,'R_PARAMS_INDEX.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
inp<-read.csv(file.path(out,'INPUTS.csv'))
dir.create(file.path(out,'R_inputs'),showWarnings=FALSE)
for(i in seq_len(nrow(inp))){
 d<-read.csv(file.path(inp$directory[i],'train.csv'));x<-as.matrix(d[grep('^x[0-9]+$',names(d))]);y<-d$y
 writeBin(as.double(cbind(x,y)),file.path(out,'R_inputs',paste0(inp$task[i],'.f64')),size=8,endian='little')
}

