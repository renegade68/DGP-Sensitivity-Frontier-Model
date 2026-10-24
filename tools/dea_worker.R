args<-commandArgs(trailingOnly=TRUE)
.libPaths(strsplit(Sys.getenv("FRONTIER_R_LIBS"),.Platform$path.sep,fixed=TRUE)[[1]])
cfg<-jsonlite::fromJSON(args[1])
stopifnot(as.character(packageVersion("Benchmarking"))=="0.33")
stopifnot(as.character(packageVersion("lpSolveAPI"))=="5.5.2.0.17.15")
source(cfg$source)
dat<-read.csv(cfg$train,check.names=FALSE);x<-as.matrix(dat[,seq_len(ncol(dat)-1),drop=FALSE]);y<-dat$y
q<-as.matrix(read.csv(cfg$query,check.names=FALSE));n<-nrow(x)
first<-instrumented_dea(x,y,q);chosen<-certificates(first,x,y,q)
initial_bad<-which(!chosen$good);source_code<-ifelse(chosen$good,1L,0L);second<-NULL
if(length(initial_bad)) {
  second<-instrumented_dea(x,y,q[initial_bad,,drop=FALSE],"none")
  c2<-certificates(second,x,y,q[initial_bad,,drop=FALSE]);selected<-initial_bad[c2$good]
  for(k in names(chosen)) {
    if(is.matrix(chosen[[k]]))chosen[[k]][selected,]<-c2[[k]][c2$good,,drop=FALSE]
    else chosen[[k]][selected]<-c2[[k]][c2$good]
  }
  source_code[selected]<-2L
}
saveRDS(list(default=first,no_scaling=second,no_scaling_query_indices=initial_bad),file.path(cfg$output,"R_attempts.rds"))
mat<-cbind(chosen$value,source_code,chosen$weights,chosen$alpha,chosen$beta)
con<-file(file.path(cfg$output,"R_certificates.f64"),"wb");writeBin(as.double(mat),con,size=8,endian="little");close(con)
meta<-list(rows=nrow(mat),columns=ncol(mat),default_unqualified_rows=initial_bad,
  unresolved_after_benchmarking=which(source_code==0L),R=R.version.string,
  Benchmarking=as.character(packageVersion("Benchmarking")),lpSolveAPI=as.character(packageVersion("lpSolveAPI")))
jsonlite::write_json(meta,file.path(cfg$output,"R_metadata.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
