
files <- list.files(pattern = 'r.csv')

total_files <- length(files)
total_files <- total_files - total_files %% 10
# print(total_files)

ONEFILE <- read.csv(files[1])

for (FILE in files[-1]){
	df <- read.csv(FILE)
	ONEFILE <- merge(ONEFILE, df, by='ExperimentName')
	pct <- which(files==FILE)
	if ((((pct/total_files)*100) %%5) ==0){
		print(paste0(pct/total_files* 100,'% Completed'))
	}
	# print(FILE)
	# print(nrow(df))
	# cat("\n")
	# cat(print(nrow(ONEFILE)))
}

rownames(ONEFILE)<-ONEFILE[,1]
# ONEFILE <- ONEFILE[,-1]
# write.csv(ONEFILE, 'onefile1.csv')
ONEFILE <- as.data.frame(t(ONEFILE[,-1]))
write.csv(ONEFILE, 'onefile.csv')

