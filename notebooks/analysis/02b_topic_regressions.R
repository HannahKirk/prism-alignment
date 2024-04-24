#setup
rm(list = ls())
library(data.table)
library(jsonlite)
library(lmtest)
library(sandwich)
library(ggplot2)
library(stargazer)

#set wd to file path
current_file_path <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_file_path)
#define paths
path_data <- "../../data/"
path_conv <- paste0(path_data,"conversations.jsonl")
path_surv <- paste0(path_data,"survey.jsonl")
path_res <- "../../results/clusters/"
path_clusters <- paste0(path_res,"opening_prompt_cluster_df_original.csv")
path_text <- paste0(path_res,"opening_prompt_text_df_original.csv")
path_output <- paste0(path_res,'topic_regression_tests.csv')

###
### Prepare data 
###

#load data
dt_clusters <- fread(path_clusters)
dt_text <- fread(path_text)
dt_conv <- data.table(stream_in(file(path_conv)))
dt_surv <- data.table(stream_in(file(path_surv)))


#process data
dt_full <- data.table::merge.data.table(
  dt_text,dt_conv[,.(conversation_id,user_id,conversation_type)],
  by.x='id',by.y='conversation_id')
dt_full <- data.table::merge.data.table(dt_full,dt_surv,by='user_id')

variables <- c("text","user_id",
               "gender","age","location.special_region",
               "ethnicity.simplified",
               "religion.simplified",
               "conversation_type",
               "cluster_id")

variables_rename <- c("text","user_id",
                      "gender","age","region",
                      "ethnicity",
                      "religion",
                      "conversation_type",
                      "cluster_id")

dt_full <- dt_full[,..variables]
setnames(dt_full,variables,variables_rename)



#set factors

dt_full$conversation_type <- relevel(factor(dt_full$conversation_type),ref='unguided')
dt_full$gender <- relevel(factor(dt_full$gender),ref='Male')
dt_full$region <- relevel(factor(dt_full$region),ref='US')
dt_full$ethnicity <- relevel(factor(dt_full$ethnicity),ref='White')
dt_full$religion <- relevel(factor(dt_full$religion),ref='No Affiliation')


###
### Analysis
###

dt_R2 <- data.table('cluster' =  dt_clusters$gpt_description, 'cluster_id' = dt_clusters$cluster_id)
dt_R2 <- dt_R2[cluster_id != -1]

#run regressions for all topics
gen_res_table <- function(dt=dt_full) {
  ### coef of interest is an index
  res_table <- data.table('cluster_id'=NA,'cluster_name'=NA,'coefficient'=NA,'estimate'=NA,'pval'=NA)
  for (i in 0:21) {
    #lpm 
    model <- lm(cluster_id== i ~gender  + age +  region + ethnicity + religion + conversation_type ,data=dt )
    
    m = summary(model)
    dt_R2[cluster_id == i,R2 := m$r.squared]
    
    res <- coeftest(model, vcov = vcovHC(model, type = "HC1",
                                         cluster = ~user_id))
    res_clean <- data.table('cluster_id' = i,
                            'cluster_name' = gsub("[^[:alpha:]]", "", dt_clusters[cluster_id == i,gpt_description]),
                            'coefficient' = rownames(res),
                            'estimate' = round(res[,'Estimate'],4),
                            'pval' = round(res[,'Pr(>|t|)'],4))
    
    res_table <- rbind(res_table,res_clean)
  }
  output <- res_table[!is.na(cluster_id) & coefficient != '(Intercept)'] #remove NA row
  output[,'significant_99pc' := fifelse(pval<0.01,1,0)]
  return(output)
}

dt <- gen_res_table()

#stats
mean(dt_R2$R2)
max(dt_R2$R2)
min(dt_R2$R2)
dt[significant_99pc==1 & (coefficient %in% c('conversation_typecontroversy guided','conversation_typevalues guided') == FALSE)       ,.N]

###
### Export
###

fwrite(dt,path_output)





