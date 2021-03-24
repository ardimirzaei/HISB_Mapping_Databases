print("Starting File Generation")
library(stringr)
library(gtools)

# setwd("C:\\Users\\amir9048\\Google Drive\\University\\Python Script\\HISB_Databases\\Experiments")
FILES <- c(
	'HINTS_COMPLETE',
	'CHIS_COMPLETE',
	'ANHCS',
	'GSS'
)

FILES_HINTS <- c(
	'HINTS_1',
	'HINTS_2',
	'HINTS_3',
	'HINTS_PR',
	'HINTS_4c1',
	'HINTS_4c2',
	'HINTS_4c3',
	'HINTS_4c4',
	'HINTS_FDA',
	'HINTS_FDA_2',
	'HINTS_5c1',
	'HINTS_5c2',
	'HINTS_5c3'
)


ANHCS <- 'ANHCS'
GSS <- 'GSS'

MODEL <- c(
	"LR",
	# "SVM",
	"RFC",
	"FC",
	"LSTM",
	"GLV-LSTM",
	"GLV-FC"
	# "HYBRID"
)


x <- 0
list_of_exp <- data.frame(exp=NA,Training=NA,Testing=NA, Analysis=NA)

# INDIVIDUAL YEAR TRAINING
for (i in 2:length(FILES_HINTS)-1){
	x <- x+1
	for (j in MODEL){
		temp = list()
		temp['exp'] <- paste0('exp001_', x ,'_IDVYEAR_' , j)
		temp['Training'] <- str_replace(str_c(paste0("'",FILES_HINTS[1:i],"',"), collapse=""),",$","")
		temp['Testing'] <- 	str_replace(str_c(paste0("'",FILES_HINTS[i+1],"',"), collapse=""),",$","")
		temp['Analysis'] <- j
		temp <- c(temp['exp'][[1]],temp['Training'][[1]],temp['Testing'][[1]],temp['Analysis'][[1]])
		list_of_exp <- rbind(list_of_exp,temp)
	}

}

x <- 0
EXP02_FILES <- permutations(n=4,r=2,v=FILES,repeats.allowed=F)
for (i in 1:nrow(EXP02_FILES)){
	x <- x+1
	for (j in MODEL){
		temp = list()
		temp['exp'] <- paste0('exp002_', x ,'_DBASE_', substr(EXP02_FILES[i,1],1,1) ,"_" , j)
		temp['Training'] <- str_replace(str_c(paste0("'",EXP02_FILES[i,1],"',"), collapse=""),",$","")
		temp['Testing'] <- 	str_replace(str_c(paste0("'",EXP02_FILES[i,2],"',"), collapse=""),",$","")
		temp['Analysis'] <- j
		temp <- c(temp['exp'][[1]],temp['Training'][[1]],temp['Testing'][[1]],temp['Analysis'][[1]])
		list_of_exp <- rbind(list_of_exp,temp)
	}

}

# Python Files
list_of_exp <- list_of_exp[-1,]

NUMBER <- paste0('_',str_pad(gsub('_',"",str_extract(list_of_exp[,'exp'],"_[0-9]{1,2}_")),2,pad=0),'_')

list_of_exp[,'exp'] <- str_replace(list_of_exp[,'exp'],"_[0-9]{1,2}_", NUMBER)


write.csv(list_of_exp, 'EXPERIMENT_LIST.csv', row.names=FALSE)


for (experiment in 1:nrow(list_of_exp['exp'])){
	exp_file <- readLines("__pyfile__01_build_models")
	#FLAGSTART#
	FLAG_START <- which(exp_file=="#FLAGSTART#")
	T <- FLAG_START + 2
	P <- FLAG_START + 3
	M <- FLAG_START + 4
	A <- FLAG_START + 5
	exp_file[T] <- paste0("TRAINING_RANGES = [",list_of_exp[experiment,'Training'],"]")
	exp_file[P] <- paste0("PREDICTION_RANGES = [",list_of_exp[experiment,'Testing'],"]")
	exp_file[M] <- paste0("MODEL_NAME = '",list_of_exp[experiment,'exp'],"'")
	exp_file[A] <- paste0("ANALYSIS_METHOD = '",list_of_exp[experiment,'Analysis'],"'")
	writeLines(exp_file, paste0(list_of_exp[experiment,1],".py"))
}

# POSITIONING <- length(MODEL)
# # Experiment 1 List PBS Files
# exp_list <- grep("exp001",list_of_exp[,'exp'])

# for (i in 0:11){
# 	pbs_file <- readLines('__pbsfile_runexp')
# 	pbs_file[3] <- paste0(pbs_file[3],'exp001_',letters[i+1])
# 	for (experiment in exp_list[(POSITIONING*i+1):(POSITIONING*i+POSITIONING)]){
# 		tmp_Add <- paste0("python3 ",list_of_exp[experiment,1], ".py > LogOutputs/", list_of_exp[experiment,1], ".output")
# 		pbs_file <- c(pbs_file, tmp_Add)
#                 tmp_Add <- paste0("mv ",list_of_exp[experiment,1], ".py PyFiles/") #, list_of_exp[experiment,1], ".output")
#                 pbs_file <- c(pbs_file, tmp_Add)

# 	}
# 	writeLines(pbs_file, paste0('exp',"001_",letters[i+1],".pbs"))
# }



################ DO NOT NEED THIS #######################
# # Experiment 2 List PBS Files
# exp_list <- grep("exp002",list_of_exp[,'exp'])

# for (i in 0:11){
# 	pbs_file <- readLines('__pbsfile_runexp')
# 	# pbs_file <- 1:POSITIONING
# 	pbs_file[3] <- paste0(pbs_file[3],'exp002_',letters[i+1])
# 	for (experiment in exp_list[(POSITIONING*i+1):(POSITIONING*i+POSITIONING)]){
# 		# print(experiment)
#     		tmp_Add <- paste0("python3 ",list_of_exp[experiment,1], ".py > LogOutputs/", list_of_exp[experiment,1], ".output")
# 	    	pbs_file <- c(pbs_file, tmp_Add)
#         	tmp_Add <- paste0("mv ",list_of_exp[experiment,1], ".py PyFiles/") #, list_of_exp[experiment,1], ".output")
# 	        pbs_file <- c(pbs_file, tmp_Add)

#     		# print(pbs_file)
# 	}
# 	writeLines(pbs_file, paste0('exp002_',letters[i+1],".pbs"))
# 	# print(paste0('exp002_',letters[i],".pbs"))
# }
############### DO NOT NEED THIS #######################




# BUILD LR MODELS 

exp_list <- grep("LR",list_of_exp[,'Analysis'])


for (experiment  in exp_list){
 	pbs_file <- readLines('__pbsfile_runexp')
	pbs_file[3] <- paste0(pbs_file[3],'exp_LR_',str_extract(list_of_exp[experiment,'exp'],"[0-9]_[0-9]{1,2}"))
	# Add python commands
	tmp_Add <- paste0("python3 ",list_of_exp[experiment,1], ".py > LogOutputs/", list_of_exp[experiment,1], ".output")
	pbs_file <- c(pbs_file, tmp_Add)
	tmp_Add <- paste0("mv ",list_of_exp[experiment,1], ".py PyFiles/") #, list_of_exp[experiment,1], ".output")
    pbs_file <- c(pbs_file, tmp_Add)

    # Write out files
    writeLines(pbs_file,paste0('exp_LR_',str_extract(list_of_exp[experiment,'exp'],"[0-9]_[0-9]{1,2}"),".pbs"))
}

# BUILD GLV-FC MODELS

exp_list <- grep("GLV-FC",list_of_exp[,'Analysis'])


for (experiment  in exp_list){
        pbs_file <- readLines('__pbsfile_runexp')
        pbs_file[3] <- paste0(pbs_file[3],'exp_GLVFC_',str_extract(list_of_exp[experiment,'exp'],"[0-9]_[0-9]{1,2}"))
        # Add python commands
        tmp_Add <- paste0("python3 ",list_of_exp[experiment,1], ".py > LogOutputs/", list_of_exp[experiment,1], ".output")
        pbs_file <- c(pbs_file, tmp_Add)
        tmp_Add <- paste0("mv ",list_of_exp[experiment,1], ".py PyFiles/") #, list_of_exp[experiment,1], ".output")
    pbs_file <- c(pbs_file, tmp_Add)

    # Write out files
    writeLines(pbs_file,paste0('exp_GLVFC_',str_extract(list_of_exp[experiment,'exp'],"[0-9]_[0-9]{1,2}"),".pbs"))
}



# BUILD OTHER MODELS 


exp_list <- grep("[^LR]",list_of_exp[,'Analysis'])


pbs_file <- readLines('__pbsfile_runexp')
pbs_file[3] <- paste0(pbs_file[3],'exp_01Models')
	# Add python commands
for (experiment  in exp_list){
	tmp_Add <- paste0("python3 ",list_of_exp[experiment,1], ".py > LogOutputs/", list_of_exp[experiment,1], ".output")
	pbs_file <- c(pbs_file, tmp_Add)
	tmp_Add <- paste0("mv ",list_of_exp[experiment,1], ".py PyFiles/") #, list_of_exp[experiment,1], ".output")
    pbs_file <- c(pbs_file, tmp_Add)

    # Write out files
}

glv_fc_lines <- grep("GLV-FC",pbs_file)
pbs_file[glv_fc_lines] <- ""

writeLines(pbs_file,'exp_01Models.pbs')
