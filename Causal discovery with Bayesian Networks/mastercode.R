

#import packages.
library(bnlearn)
library(MASS)
library(tidyverse)
library(CARRoT)
library(data.table)
library(gRbase)







#load rda file of networks from bnlearn repository.

load("asia.rda")
load("survey.rda")
load("sachs.rda")
load("child.rda")
load("earthquake.rda")
load("alarm.rda")



#generate cpdag of loaded network.
d=cpdag(bn)


library(graph)
library(igraph)
#plot cpdag.
plot(as_graphnel(as.igraph(d)))


#detach packages because they had a conflict with Rbase library.
detach("package:gRbase", unload=TRUE) 
detach("package:igraph", unload=TRUE)


#adjacency matrix of loaded network.
true_matrix=amat(bn)


#Direct ancestral relatoin matrix for loaded network.
true_matrix_1=solve(diag(11)-(true_matrix))
true_matrix_1=1*apply(true_matrix_1, 2, function(x) x>=1)


#nr of nodes in network.
nr_nodes=11
#maxparentsize for scorefile.
max_parent_size=4


#find how many values each variable in network takes by generating a large dataset.
data <- rbn(x = bn, n = 100000)

data_set_large=as.data.frame(map_df(data, as.numeric))

p=lapply(1:nr_nodes, function(x){unique(data_set_large[,x])})


############################################################################################################
#Function:write_func
############################################################################################################
#Input:
#read_this: path of scorefile containing P(D|G).

#nr_nodes:how many nodes in network.

#paramter:how many parentcombinations for each node.

#name:what name chould be given to new scorefile containing P(G,D).

#nr_data:how many data samples is scorefile based on

#Output:scorefile containing scores P(G,D)


#This function adds prior for graph G added on the P(D|G) contained in scorefile , in order to estimate P(G,D).
#############################################################################################################
write_func=function(read_this,nr_nodes,parameter,name,t,nr_data){
  
  # These rows in scorefile are excluded sinse these rows contains node together with how many parent-combination of that node. 
  seqq=c(seq(1,nr_nodes*parameter,parameter)+1:nr_nodes,1)
  
  #Add prior on rest of the rows.
  for(i in setdiff(1:nrow(read_this),seqq)){
    component=as.numeric(strsplit(read_this[i,]," ")[[1]])
    component[1]=component[1]-(1+t)^(length(component)-2)*log(nr_data)
    
    read_this[i,]=gsub(",","",toString(component))
    
  }
  
  #convert scorefile back to type .score
  colnames(read_this)=NULL
  
  
  
  read_this_name=paste0('C:/Users/rasyd/Documents/gitrepo/master/score_folder/scores/csi_sachs/n200/t05/',"temp.",name,".score")
  #write new scorefile
  write.matrix(read_this,sep=" , ",file=read_this_name)
  
  
  
  
}



############################################################################################################
#Function:run_write
############################################################################################################
#Input:
#nr_of:how many scorefiles should be made.

#nr_nodes:how many nodes in network.

#paramter:how many parentcombinations for each node.

#t:tuning paramter for prior.

#nr_data:how many data samples is scorefile based on

#Output:scorefiles containing scores P(G,D)


#This function adds prior for multiple scorefiles where how many scorefiles are  specified by nr_of.
#############################################################################################################


run_write=function(nr_of,parameter,nr_nodes,t,nr_data){
 
  for(j in 1:nr_of){
    add=samp[j]
    pas_string=toString(add)
    if(k==1){
      score_type=paste0("cat","type",pas_string)}
    if(k==2){
      score_type=paste0("csi","type",pas_string)
    }
    read_this=paste0('C:/Users/rasyd/Documents/gitrepo/master/score_folder/scores/csi_sachs/n200/',"temp.",score_type,".score")
    read_this=read.csv(file = read_this, header = FALSE)
    
    write_func(read_this,nr_nodes,parameter,score_type,t,nr_data )
    
  }
  
  
  
}

# Input variables for prior function write_func. 
#Index names for score files.
samp=3:22
#If k=1 scorefile name is of cattype if k=2  scorefile name is of csitype.
k=1
#Number of parent combinations for each node.
parameter=386
#nr_nodes assigns how many nodes in network.
nr_nodes=11

#Tuning paramter of prior.
t=0.5
#Size of the data the scores are calculated from.
nr_data=200
#Add prior on 20 score files.
run_write(20,parameter,nr_nodes,t,nr_data)


############################################################################################################
#Function:run_func_1
############################################################################################################
#Input:
#max_parent_size:how many scorefiles should be made.

#nr_nodes:how many nodes in network.

#paramter:how many parentcombinations for each node.

#j:iterator index.

#p:contains a list of lists where each list contains the values of a node in the network

#k:is a vector used to set name on scorefile depending on if the scorefile is CSI type or CPT type


#Output:scorefiles containing scores P(D|G)


#This function calculates the scorefiles.
#############################################################################################################

 
run_func_1=function(max_parent_size,nr_nodes,j,p,k){
  
  #generate data of size ns[i] from loaded network
  data <- rbn(x = bn, n = ns)
  #convert variables from factor to numeric
  data_set=as.data.frame(map_df(data, as.numeric))
  #convert to data.table
  setDT(data_set)
  
  
  add=j+samp
  pas_string=toString(add)
  
  #based on k scorefile name changes.This function was inspired by an earlier function these if statements are not necessary
  if(k[1]==1){
    score_type_1=paste0("cat","type",pas_string)}
  if(k[2]==2){
    score_type_2=paste0("csi","type",pas_string)
  }
  
  #calulate CPT-scorefile
  csitree_calc_parent_score_to_file_2(data_set,score_type_1 , max_parent_size, file_out="temp",p)
  #calculate CSI-scorefile
  csitree_calc_parent_score_to_file_3(data_set,score_type_2 , max_parent_size, file_out="temp",p)
  
}


############################################################################################################
#Function:run_func_2
############################################################################################################
#Input:
#max_parent_size:how many scorefiles should be made.

#nr_nodes:how many nodes in network.

#true_matrix:some transformation of the adjacency matrix of the network

#j:iterator index.

#p:contains a list of lists where each list contains the values of a node in the network

#cpt_or_csi:which scoretype to run MCMC on.


#Output:AUC


#This runs MCMC over a scorefile 
#############################################################################################################


run_func_2=function(max_parent_size,nr_nodes,j,p,true_matrix,cpt_or_csi){
 
  add=j+samp
  pas_string=toString(add)
  if(cpt_or_csi==1){
    score_type=paste0("cat","type",pas_string)}
  if(cpt_or_csi==2){
    score_type=paste0("csi","type",pas_string)
  }
 
  #read scorefiles
  read_this=paste0('C:/Users/rasyd/Documents/gitrepo/master/score_folder/scores/csi_sachs/t2/',"temp.",score_type,".score")
  read_this=read.csv(file = read_this, header = FALSE)
  #run MCMC
  MCMC_AUC=func_MCMC(read_this,true_matrix,max_parent_size,j,nr_nodes,ns,cpt_or_csi)
  
  return(MCMC_AUC)
}


#Adding length of previous run.  
samp=1

#Set data size.
ns=1000
#How many data files.
n=20
#Which score type.
k=c(1,2)

cpt_or_csi=1

#Run in parallel.
#Import packages needed for parallel runs.
library(parallel)
library(doSNOW)
#Divide processor into smaller clusters.
cl <- makeCluster(6,type = "SOCK",outfile="log.txt")
registerDoSNOW(cl)




  #Feed clusters functions the input variables needed.
  clusterExport(cl,c("func","func_2","samp","bn","k","cpt_or_csi","p","run_func_1","run_func_2","cat_calc_parent_score_to_fil_3_plane","CSI_tree_apply_imp_3_mat_B_3","csitree_calc_parent_score_to_file_3","csitree_calc_parent_score_to_file_2","func_MCMC","ns","max_parent_size","nr_nodes","true_matrix_1","n"),envir = environment())

  
  
  
  #import libraries for each cluster.
  clusterEvalQ(cl, c(library(data.table),library(MASS)
                     ,library(tidyverse)
                     ,library(CARRoT),library(bnlearn),library(gRbase)))
  
  
  #calculate scorefiles.
  parLapply(cl,1:n,function(x){run_func_1(max_parent_size,nr_nodes,x,p,k)})
  
  #Run MCMC on them and return AUC vector of all runs either for csi-score or for cpt score.
  AUC_vec=unlist(parLapply(cl,1:n,function(x){run_func_2(max_parent_size,nr_nodes,x,p,true_matrix_1,cpt_or_csi)}))
  
  


# Stop cluster on master
stopCluster(cl)





#All plots are generated with a similar construction to the following code
#make an empty list where every element in the list will be a matrix.There will be four matrixes each one containing the result of CSI score results using MCMC,
#CSI score result using  exact algortihm,CPT score results using  MCMC and CPT score results using exact algortihm.
#Every colum in the matrices represents the result for a specific data sample size
Plot_list=list()
#we will generate 20 AUC for every data sample size
#set k=1
k=1
#set m=20
m=20
# ns  defines the domain of data sample sizes
ns=c(1000,500,200)
#define empty AUC vector
AUC_calc=matrix(0,20,1)
#for 4 different results
for(j in 1:4){
#for all sample sizes
  for(i in 1:length(ns)){
    #if CSI score AUC results using MCMC
    if(j==1){
      #extract names of all matrices
      matrix_name_extract=list.files(path="C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/csi_sachs/direct_cause/joint", pattern=NULL, all.files=FALSE,
                                     full.names=FALSE)
      #sort them on modified time 
      details=file.info(dir(path = "C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/csi_sachs/direct_cause/joint", full.names = TRUE), extra_cols = FALSE)
      details = details[with(details, order(as.POSIXct(mtime))), ]
     
     
      
      #extract matrices from k:m
      paste_to_each=rownames(as.data.frame(details))[k:m]
      #define matrix paths
      paste_to_each=lapply(1:length(paste_to_each[k:m]),function(x)gsub(" ", "", paste_to_each[x]))
      
      print(paste_to_each)
    }
    #if CPT score AUC results using MCMC
    if(j==2){
      
      
      
      matrix_name_extract=list.files(path="C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/cat_sachs/direct_cause/joint", pattern=NULL, all.files=FALSE,
                                     full.names=FALSE)
     
      
      
        match_sring=grep(toString(ns[i]),matrix_name_extract)
        matrix_name_extract=matrix_name_extract[match_sring]
       
      
      
      
      matrix_name_extract=sort(matrix_name_extract,decreasing = TRUE)
      
      paste_to_each=paste("C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/cat_sachs/direct_cause/joint/",matrix_name_extract)
      
      
      paste_to_each=lapply(1:length(paste_to_each[k:m]),function(x)gsub(" ", "", paste_to_each[x]))
      
      
    }  
    
    #if CSI score AUC results using exact algorithm
    if(j==3){
     
        matrix_name_extract=list.files(path="C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/exact_csi_sachs/t0", pattern=NULL, all.files=FALSE,
                                       full.names=FALSE)
  
        
        
        paste_to_each=paste("C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/exact_csi_sachs/t0/",matrix_name_extract)
        
        paste_to_each=paste_to_each[order(as.numeric(gsub("[^0-9]+", "", paste_to_each)))]
        
        
        paste_to_each=lapply(1:length(paste_to_each),function(x)gsub(" ", "", paste_to_each[x]))[k:m]
        print( paste_to_each)
      
    }
    
    
    #if CPT score AUC results using exact algorithm
    if(j==4){
      
      matrix_name_extract=list.files(path="C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/exact_cat_survey", pattern=NULL, all.files=FALSE,
                                     full.names=FALSE)
      
      
      
      paste_to_each=paste("C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/exact_cat_survey/",matrix_name_extract)
      
      paste_to_each=paste_to_each[order(as.numeric(gsub("[^0-9]+", "", paste_to_each)))]
      
      paste_to_each=lapply(1:length(paste_to_each),function(x)gsub(" ", "", paste_to_each[x]))[k:m]
      
      
      
    }
    
    
    #Extract matrices
    Matrix_vec=lapply(1:length(paste_to_each),function(x)unname(as.matrix(read.csv(file = paste_to_each[[x]], header = FALSE))))
    
    #if matrices is from exact algorithm matrix elements becomes strings.We have to convert them to numeric matrices
    if(j>2){
      Matrix_vec=lapply(1:length(paste_to_each),function(x)Matrix_vec[[x]][-1,])
      
      
      Matrix_vec=lapply(1:length(paste_to_each),function(x){matrix(as.numeric(Matrix_vec[[x]]),ncol = ncol(Matrix_vec[[x]]))})
      
      
      
      
      
    }
  
    
    #use class_prob to get matrix of two columns.First column reprsents estimated direct ancestral relation probabilities sorted in a decending way.Second column 
    #represents true direct ancestral relation matrix sorted in the same order as estimated probabilities. 
    AUC_input=lapply(1:length(Matrix_vec),function(x)class_prob(Matrix_vec[[x]])) 
    
    #concatinate zero vector with AUC of every estimated matrix
    AUC_calc=cbind(AUC_calc,do.call(rbind,lapply(1:length(Matrix_vec),function(x)AUC(AUC_input[[x]][,1], AUC_input[[x]][,2]))))
    
    k=k+20
    m=m+20
  }
  #Fill in matrices into list
  Plot_list[[j]]=AUC_calc
  k=1
  m=20
  #redefine zero vector for each j
  AUC_calc=matrix(0,20,1) 
  
}







#remove zero vector from all matrices
Plot_list=lapply(1:length(Plot_list),function(x) as.matrix(Plot_list[[x]][,-1]))


#import ggplot2
library(ggplot2)



#Rechape every matrix in Plot_list and concatinate it with indicator of which column in belonged to
box_matrix=lapply(1:4,function(z){Reduce("rbind",
                                         Reduce("rbind",lapply(1:ncol(Plot_list[[z]]),function(y){lapply(1:nrow(Plot_list[[z]]),
                                                                                                         function(x){cbind(Plot_list[[z]][x,y],y*as.numeric(x>0))})})))})
#give name to every element in box_matrix
lapply(1:length(box_matrix),function(x)colnames(box_matrix[[x]])<<-c("AUC","ns"))


#if sorting was done wrong reshuffle at the scal x axis correctely
for(i in 1:4){
  if(i==1){
     ns=c(1000,500,200)
    
    
   
    
    apply(matrix(1:nrow(box_matrix[[i]]),1,nrow(box_matrix[[i]])),2,function(x){box_matrix[[i]][x,2]<<-ns[box_matrix[[i]][x,2]]})
  }
  if(i==2){
    ns=c(1000,500,200)
    
    
    apply(matrix(1:nrow(box_matrix[[i]]),1,nrow(box_matrix[[i]])),2,function(x){box_matrix[[i]][x,2]<<-ns[box_matrix[[i]][x,2]]})
  }
  
  if(i==3){
    ns=c(1000,500,200)
   
    apply(matrix(1:nrow(box_matrix[[i]]),1,nrow(box_matrix[[i]])),2,function(x){box_matrix[[i]][x,2]<<-ns[box_matrix[[i]][x,2]]})
  }
  
  if(i==4){
    ns=c(500,200,1000)
    
    
    apply(matrix(1:nrow(box_matrix[[i]]),1,nrow(box_matrix[[i]])),2,function(x){box_matrix[[i]][x,2]<<-ns[box_matrix[[i]][x,2]]})
    
  }
}

#assign each plot to a variable
for(i in 1:length(box_matrix) ){
  token_seq=paste("token_",i,"")
  
  assign(token_seq,ggplot(as.data.frame(box_matrix[[i]]), aes(x=(samples=factor(ns)), y=AUC,color=ns)) + labs(y= "AUC", x = "ns") 
         +geom_boxplot(outlier.color="black"))
}

library(gridExtra)
#plot a grid-plot of all box-plots
grid.arrange(`token_ 1 `,`token_ 2 `,`token_ 3 `,`token_ 4 `)







#function for generating Roc curves
compute_roc=function(compare,i){
  #list of colors of plot
  color=c("red","blue","yellow","brown")
  
  #flatten matrix to vector
  rechape_compare=c(compare)
  #attain order of  rechape_compare
  rechape_compare_order=order(rechape_compare,decreasing = TRUE)
  
  #order rechape_compare in descending order
  order_MCMC=rechape_compare[rechape_compare_order]
  
  #flatten true ancestral relation matrix
  rechape_true=c(true_matrix)
  #sort it in the same oder as rechape_compare
  order_true=rechape_true[rechape_compare_order]
  
  #concatinate vectors
  compare_true_with_mcmc=cbind(order_MCMC,order_true)
  #count how many 1's and 0's in order_true
  how_many_total=table(order_true)
  
  #count how many 1's and zeros exist every time one adds element from from  order_true
  how_many_one_zero=lapply(1:nrow(compare_true_with_mcmc),function(x)table(order_true[1:x]))
  
  #how_many_one_zero is a list transforming it into a matrix
  how_many_one_zero_rbind=do.call(rbind,how_many_one_zero)
  
  #how many ones are in order_true
  how_many_total=matrix(how_many_total)[,1]
  
  #when number of 0's is 0 in first rows in how_many_one_zero_rbind.On these rows in  how_many_one_zero_rbind number of 1's get duplicated 
  #therefore one had to manually set 0 on these rows.In addition how_many_one_zero_rbind changes the placement of what is count of zero and what is count of 1.
  how_many_one_zero_rbind=t(apply(how_many_one_zero_rbind, 1, function(x)if(sum(x)<= how_many_total[2]){ sort(replace(x, duplicated(x), 0))}else{x}))
  
  
  #calculate TPR and FPR
  how_many_one_zero_rbind_transform=apply(matrix(1:ncol(how_many_one_zero_rbind),1,ncol(how_many_one_zero_rbind)), 2, 
                                          function(x) how_many_one_zero_rbind[,x]/how_many_total[x])
  
  #create Roc curve
  lines(how_many_one_zero_rbind_transform[,1],how_many_one_zero_rbind_transform[,2],type = "l",col=color[i])
  
}
#import matrices
call_on_al_matrix=read.csv("C:/Users/rasyd/Documents/gitrepo/master/BIDA/matrix1.txt",sep=",",header = TRUE)
call_on_al_matrix_2=read.csv("C:/Users/rasyd/Documents/gitrepo/master/BIDA/matrix.txt",sep=",",header = TRUE)
#plot empty plot
plot(NA, type="n", xlab="", ylab="", xlim=c(0, 1), ylim=c(0, 1))

#draw Roc curves
compute_roc(edge_matrix_1_cat,1)
compute_roc(edge_matrix_1,2)
compute_roc(as.matrix(call_on_al_matrix),3)
compute_roc(as.matrix(call_on_al_matrix_2),4)
legend("bottomright", cex=0.5, title="ROC-Curve",
       c("MCMCcat","MCMCSI","Exact_cat","Exact_csi"), fill=c("red","blue","yellow","brown"), horiz=TRUE)













############################################################################################################
#Function:cat_calc_parent_score_to_fil_3_plane
############################################################################################################
#Input:
#data:data used.

#node:which node to calculate log-marginal likelihood from.

#parent_comb:Specific parent combination



#p:contains a list of lists where each list contains the values of a node in the network




#Output:Return log-marginal-likelihood


#This function calculates the CPT-log-marginal-likelihood
#############################################################################################################


cat_calc_parent_score_to_fil_3_plane <- function(data,node,parent_comb,p){	
  
  
  
  
  # N  in the BDEU prior is set to 1
  N <-1 

  
  #Function for counting specific configuration in data
  M_xi_parent_count=function(data,col_2){
    
    M_xi_parent=data[, .(n = .N), by = col_2]
    return(M_xi_parent)
  }
  
  
  #function for comparing vector with row in matrix
  comparetorow=function(x,y){
    nr=nrow(x)
    nc=ncol(x)
    ret=!vector("logical",nr)
    for(i in 1:nr){
      for(k in 1:nc){
        if(x[i,k]!=y[k]){
          ret[i]=FALSE
          break
        }
      }
    }
    return(ret)
  }
  
  
  
  
  
  
  
  
  
  #List of unique valuesof node
  uniq_Xi_value=matrix(p[[node]])
  
  
  #nr of unique values for node
  nr_uniq_Xi_value1=nrow(uniq_Xi_value)
  
  #parents of node
  par=parent_comb
  
  #columnname of node and nodes parents in dataset 
  col_2=names(data[,.SD,.SDcols=c(node,par)])
 
  #part of data with nodes parents as columnname
  parent_set_entries=data[,.SD,.SDcols=c(par)] 
  
  
  #unique configurations in dataset for parent_set_entries
  uniq_par_value=as.matrix(unique(parent_set_entries))
  
  
  #number of unique parent_set_enteries in dataset
  nr_uniq_par_value=nrow(unique(parent_set_entries))
  
  
  #list of values of parents to node
  parent_set_entries_to_alpha=p[c(par)]
  
  #multiplication of length of all element in list  parent_set_entries_to_alpha
  in_between_move=lapply(1:length(parent_set_entries_to_alpha),function(x){length(parent_set_entries_to_alpha[[x]])})
  nr_uniq_par_value_to_alpha=Reduce('*',in_between_move)
  
  #calculate BDEU prior
  alpha_node_parnode=N/(nr_uniq_Xi_value1*nr_uniq_par_value_to_alpha)
  
  #Vector containing BDEU prior for all values of node
  alpha_node_parnode_vec=rep(alpha_node_parnode,nr_uniq_Xi_value1)
  
  #Sum of all element in alpha_node_parnode_vec
  alpha_sum_parnode=sum(alpha_node_parnode_vec)
  
  
  #count how many times every configuration in data set reduced to entries for columnnames col_2
  a=unname(as.matrix(M_xi_parent_count(data,col_2)))
  
  # CPT score set to 0
  src=0
  
  
  #For every parent configuration of node 
  for(parent in 1:nr_uniq_par_value){
    
    
    

    #extract part of matrix a that contain the parent configurations of node
    if(nrow(a)==1){
      
      b=matrix(a[,-c(1,ncol(a))],nrow = 1)
      
    }else{
      if(is.null(nrow(a[,-c(1,ncol(a))]))){
        b=matrix(a[,-c(1,ncol(a))],ncol = 1)}else{
          b=(a[,-c(1,ncol(a))])
        }
      
    }
    
    #find index in matrix a that matches specific parent configuration.This returns a logical vector for indexes of counts of how many times values of node appear 
    #for the specific parent config 
    w=a[comparetorow(b,uniq_par_value[parent,] ),]
    
    
  #if w contains only one row and is a vector
    if(is.null(nrow(w))){
      
      #make zero vector
      fill=rep(0,nr_uniq_Xi_value1)
     
      #collect counts from w
      M_X_split=(w[c(ncol(a))])
      
      # collect values of node in data from w
      con=w[c(1)]
      
     
      #code below is written to have fixed length on count vector no matter how many counts exist for node
      
      #match with theoretical values of node
      where_in_total=match(uniq_Xi_value,con)
      #remove NA from match
      where_in_total=which(where_in_total>0)
      
      #put counts in zero matrix
      fill[where_in_total]=M_X_split
      #rename zero vector
      M_X_split=fill
      #sun all elements in M_X_split
      M_parent_count=sum( M_X_split)
      
      
    }else{ 
      #Else if w contains more then one row and is a matrix.Same procedure is done as when w is a vector
      
      M_X_split=(w[,c(ncol(a))])
      
      con=w[,c(1)]
      
      
      fill=rep(0,nr_uniq_Xi_value1)
      
      where_in_total=match(uniq_Xi_value,con)
      
      where_in_total=which(where_in_total>0)
      
      
      fill[where_in_total]=M_X_split
      
      M_X_split=fill
      
      M_parent_count=sum(M_X_split)
     
    }
   
    #if some counts are different from zero exist in zero vector fill
    if(sum( M_X_split)!=0){
      
    
      #add to CPT log-marginal likelihood
      
      src=src+(lgamma(alpha_sum_parnode)-lgamma(alpha_sum_parnode+M_parent_count)+sum(lgamma(alpha_node_parnode_vec+(M_X_split))-lgamma(alpha_node_parnode_vec)))
      
    }
  }
  
  
  #return CPT log-marginal likelihood
  return(src)
}







#This function calculates CSI-log-marginal likelihood.




############################################################################################################
#Function:CSI_tree_apply_imp_3_mat_B_3
############################################################################################################
#Input:
#data:data used.

#node:which node to calculate log-marginal likelihood from.

#parent_comb:Specific parent combination



#p:contains a list of lists where each list contains the values of a node in the network




#Output:Return log-marginal-likelihood


#This function calculates the CSI-log-marginal-likelihood
#############################################################################################################


CSI_tree_apply_imp_3_mat_B_3 <- function(data, parent_set,intended_for,p){
  #set N in BDEU prior to 1
  N <-1
 
  #set containing node=intended_for together with its parentset
  #set <- c(parent_set,intended_for)
  
 #function for counting number of time configurations occure in dataset
  M_xi_parent_count=function(data,cols_2){
    
    
    
    M_xi_parent=data[, .(n = .N), by = cols_2]
    return(M_xi_parent)
  }
  
  #function for comparing vector with row in matrix
  comparetorow=function(x,y){
    nr=nrow(x)
    nc=ncol(x)
    ret=!vector("logical",nr)
    for(i in 1:nr){
      for(k in 1:nc){
        if(x[i,k]!=y[k]){
          ret[i]=FALSE
          break
        }
      }
    }
    return(ret)
  }
  
  #this function is used when selecting the root 
  indicator_function_2=function(row,ma,a){
    row_ma=ma[row,]
    row_config=row_ma[!is.na(row_ma)]
    
    

    
    if(nrow(a)==1){
      
      b=matrix(a[,-c(1,ncol(a))],nrow = 1)
      
    }else{
      if(is.null(nrow(a[,-c(1,ncol(a))]))){
        b=matrix(a[,-c(1,ncol(a))],ncol = 1)}else{
          b=(a[,-c(1,ncol(a))])
        }
      
    }
    
    
    w=a[comparetorow(b,row_config ),]
    
    M_X_split=0
    
    if(is.null(nrow(w))){
      
      fill=rep(0,length(val_intended_for))
      
      
      M_X_split=(w[c(ncol(a))])
      con=w[c(1)]
      where_in_total=match(val_intended_for,con)
      where_in_total=which(where_in_total>0)
     
      fill[where_in_total]=M_X_split
      
      M_X_split=fill
      
      
    }else{
      
      
      M_X_split=(w[,c(ncol(a))])
      con=w[,c(1)]
      
      
      fill=rep(0,length(val_intended_for))
      
      where_in_total=match(val_intended_for,con)
      #where_in_total=where_in_total[!is.na(where_in_total)]
      where_in_total=which(where_in_total>0)
      fill[where_in_total]=M_X_split
      #print(fill)
      M_X_split=fill
      
      
    }
    
    
    
    return( M_X_split)
  }

  #this function is used after root is selected.Difference between indicator_function_3 and indicator_function_2 is the first line in code when defining row_ma
  indicator_function_3=function(row,ma,a,next_element_con){
    row_ma=ma[row,-1]
    row_config=row_ma[!is.na(row_ma)]
    row_config=as.numeric(c(row_config,next_element_con))
    
    
    
    
  
    
    if(nrow(a)==1){
      
      b=matrix(a[,-c(1,ncol(a))],nrow = 1)
      
    }else{
      if(is.null(nrow(a[,-c(1,ncol(a))]))){
        b=matrix(a[,-c(1,ncol(a))],ncol = 1)}else{
          b=(a[,-c(1,ncol(a))])
        }
      
    }
    
    
    
    
    w=a[comparetorow(b, row_config),]
    
    
    M_X_split=0
    
    if(is.null(nrow(w))){
      
      fill=rep(0,length(val_intended_for))
      
      
      M_X_split=(w[c(ncol(a))])
      con=w[c(1)]
      where_in_total=match(val_intended_for,con)
      where_in_total=which(where_in_total>0)
     
      fill[where_in_total]=M_X_split
      
      M_X_split=fill
      
      
    }else{
      
      
      M_X_split=(w[,c(ncol(a))])
      con=w[,c(1)]
      
      
      fill=rep(0,length(val_intended_for))
      
      where_in_total=match(val_intended_for,con)
      
      where_in_total=which(where_in_total>0)
      fill[where_in_total]=M_X_split
   
      M_X_split=fill
      
      
    }
    
    
    
    return(M_X_split)
  }
  
  
  
  
  
  #indicator for root selection
  indicator=0
  # n is set to 0
  n=0
  
  

  
  #CSI-log-marginal-likelihood is set to 0
  sco=0
  

  
  
  #list of theoretical values of node 
  val_intended_for=p[[intended_for]]
  #how many values in val_intended_for
  len_val_intended_for=length(val_intended_for)
  
  #while TRUE
  while (n<1 ) {
    
  
    
    #if indicator is 0
    if(indicator==0){
      
      #sort element of parentset
      elements=parent_set
     
      #Renaming of len_val_intended_for
      no_split_uniq=len_val_intended_for
      
      #count how many entries of value of node exist in dataset
      M_X_no_split=table(data[,.SD,.SDcols=c(intended_for)])
      
      # The specific values of node that exist in dataset
      con=as.numeric(names(M_X_no_split))
      #Create zero vector for storing counts of occurrences of value of node
      fill=rep(0,length(val_intended_for))
     
      #this is done to have fixed length on count vector no matter how many counts exist for node
      
     
      
      
  
      
      
     
      
      #match with theoretical values of node
      where_in_total=match(val_intended_for,con)
      #remove NA from match
      where_in_total=where_in_total[!is.na(where_in_total)]
      #put counts in zero matrix
      fill[where_in_total]=as.numeric(M_X_no_split)
      #rename zero vector
      M_X_no_split=fill
      
      
      
      #calculate BDEU prior when no parent is included
      
      alpha_no_split= N/(no_split_uniq)
      alpha_no_split_vec=rep(alpha_no_split,len_val_intended_for)
      
      aplha_sum_no_split=sum(alpha_no_split_vec)
      
      
      M_sum_no_split= sum(M_X_no_split)
      #calculate log marginal likelihood for when node has no parent
      src_no_split=lgamma(aplha_sum_no_split)-lgamma(aplha_sum_no_split+M_sum_no_split)+sum(lgamma(alpha_no_split_vec+M_X_no_split)-lgamma(alpha_no_split_vec))
      
     #if input parentset is empty break whileloop return CSI-score
      if(is.null(parent_set)){
        sco=src_no_split
        
        break
      }
      
      
      
      
      #Set comparing value to 0
      split_t=0
      #Set CSI-score to 0
      scores=0
      
      #for each element in parentset of node
      for(element in (elements)){
       
        # list of columnnames for node=intended_for and element in dataset
        cols=names(data[,.SD,.SDcols=c(intended_for,c(element))])
        
        
        
        
        
        #Compute how many theoretical values does element have
        parent_set_entries_to_alpha=p[c(element)]
        
        #calculate number of unique values for element
        in_between_move=lapply(1:length(parent_set_entries_to_alpha),function(x){length(parent_set_entries_to_alpha[[x]])})
        nr_uniq_par_value_to_alpha=Reduce('*',in_between_move)
        
        
        
        
        
       # unique values of element in dataset
        split_uniq_par_val=unname(as.matrix(unique(data[,.SD,.SDcols=c(element)])))
        #number of unique values of element in dataset
        split_uniq_parent=nrow( split_uniq_par_val)
        con_2=split_uniq_par_val
        
        #initialize empty score vector
        src_split=rep(0,split_uniq_parent)
        #count how many instances of different configurations for node=intended_for and element that exist in dataset  
        c=unname(as.matrix(M_xi_parent_count(data,cols)))
        
        
        #for each value of element 
        for(row in 1:nrow(split_uniq_par_val)){
          #element value held fixed vary values of nodes,put each count into a vector M_X_split
          M_X_split=indicator_function_2(row,split_uniq_par_val,c)
          
          #if at least one configuration exist
          if (sum(M_X_split)!=0){
            
            
            alpha_split=N/(len_val_intended_for*nr_uniq_par_value_to_alpha)
            alpha_split_vec=rep(alpha_split,len_val_intended_for)
            
            alpha_sum_split=sum(alpha_split_vec)
            M_sum_split=sum(M_X_split)
            #calculate log marginal likelihood for node when element's value is held fixed
            src_split[row]=lgamma(alpha_sum_split)-lgamma(alpha_sum_split+M_sum_split)+sum(lgamma(alpha_split_vec+M_X_split)-lgamma(alpha_split_vec))
            
          }
          
          
          
          
        }
        
        #if at least one log marginal likelihood exist
        if(sum(src_split)!=0){
          #if the sum of log-marginal likelihood is greater then split_t 
          if((sum(src_split)-sum(src_no_split))>split_t ){
            #save element
            choice=element
            
            #Which value of element has  log-marginal likelihood 0
            w_zeo=which(src_split==0)
            if(length(w_zeo)!=0){
              #exclude this score
              scores=src_split[-c(w_zeo)]
              #Exclude the value related to that score
              con_3=c(con_2[-c(w_zeo),])
            }else{
              
              #else all values has a score
              scores=src_split
              
              con_3=c(con_2)
            }
            #set positive difference to be new comparing value
            split_t=(sum(src_split)-sum(src_no_split))
            
          }
        }
      }
      
      
      
      
      
      #if all element and there values have been looked at and score still is zero no root is found
      if(sum(scores)==0){
        
        sco=src_no_split
        
        break
        
      }else{
        
        #Else root is found define tree 
        
        
        #chosen elements value in data
        choise_uniq=con_3
        #number of values of element in data
        choise_unique_nr=length(choise_uniq)
        
        #matrix containing score for every branch(value)
        mat=matrix(c(scores,choise_uniq),ncol = 2)
        #element  corresponding value in mat
        al_matrix=matrix(rep(choice,choise_unique_nr),ncol = 1)
        
        #set indicator to 1
        indicator=1
        
      }
    }else{
      
      
      #continue building tree in the same way 
      scores=0
      split_t=0
      #for element in parentset
      for(element in elements){
        
        #for every row in mat
        for(row in 1:nrow(mat)){
          
          
          
          
         #go through every row in al_matrix
          parent_elements_row=al_matrix[row,]
          
          parent_elements_row= (parent_elements_row[!is.na( parent_elements_row)])
          
          #add element in branch(row)
          cols=names(data[,.SD,.SDcols=c(intended_for,parent_elements_row,element)])
          
          #if element is not in row of al_matrix continue 
          `%!in%` <- Negate(`%in%`)
          if(element%!in%parent_elements_row){
            
            #do the same procedure as described above to calculate log-marginal-likelihood of branches adding element to each row in al_matrix on every branch(row)
            #that does not contain element look for which element added to which branch gives the greatest improvement 
            
            parent_set_entries_to_alpha=p[c(parent_elements_row,element)]
            
            
            in_between_move=lapply(1:length(parent_set_entries_to_alpha),function(x){length(parent_set_entries_to_alpha[[x]])})
            nr_uniq_par_value_to_alpha=Reduce('*',in_between_move)
            
            
            split_uniq_parent_config=nrow(unique(data[,.SD,.SDcols=c(parent_elements_row,element)]))
          
            c=unname(as.matrix(M_xi_parent_count(data,cols)))
            
            
            next_el=unname(as.matrix(unique(data[,.SD,.SDcols=element])))
            con_2=next_el
            
            s_vec=rep(0,nrow(next_el))
           
            
            for(s in 1:length(s_vec)){
              
              M_X_split=indicator_function_3(row,mat,c,next_el[s])
              
              
              
              if(sum(M_X_split)!=0){
                
                
                M_sum_split=sum(M_X_split)
                
                alpha_split= N/(len_val_intended_for* nr_uniq_par_value_to_alpha)
                alpha_split_vec=rep(alpha_split,len_val_intended_for)
                
                alpha_sum_split=sum(alpha_split_vec)
                s_vec[s]=lgamma(alpha_sum_split)-lgamma(alpha_sum_split+M_sum_split)+sum(lgamma(alpha_split_vec+M_X_split)-lgamma(alpha_split_vec))
               
              }
              
              
            }
            
            
            if(sum(s_vec)!=0){
              src_no_split=mat[row,1]
              
              diff=sum(s_vec)-src_no_split 
              
              
              
              
              if(diff>split_t){
                w_zeo=which(s_vec==0)
                if(length(w_zeo)!=0){
                  scores=s_vec[-c(w_zeo)]
                  
                  con_3=c(con_2[-c(w_zeo),])
                }else{
                  scores=s_vec
                  
                  con_3=c(con_2)
                }
                
                which_element_choosen=element
               
                which_row_branch=row
                #update split_t the same way as before
                split_t=diff
                
                
              }
            }
            
          }
          
        }
      }
      
      
      #if no element added to each branches give any improvement break 
      if(all(scores==0)){
       
        break
      }else{
        
        #else add the element to the branch that gave the best improvement 
        
        #values of chosen element
        choosen_element_uniq_val=con_3
        
        #number of values of element in dataset 
        choosen_element_uniq_val_nr=length(choosen_element_uniq_val)
        
        
        
        
        #if more then one value exist for element in dataset and expand  the rows with the number of values of element-1 
        if(length(con_3)>1){
          v=rep(1,length(mat[,(ncol(mat))]))
          v[which_row_branch]=choosen_element_uniq_val_nr
          
         
          mat=(mat[rep(1:nrow(mat), times = v),])
          
          
          al_matrix=al_matrix[rep(1:nrow(al_matrix), times = v),]
          
        }
        
        #else only one value exist for element in dataset and expand only the columns
        
        #new configurations to add to mat
        new_node_value=matrix(rep(NA,nrow(mat)),ncol=1)
        
        #put values of element into new_node_value
        new_node_value[which_row_branch:(which_row_branch+choosen_element_uniq_val_nr-1),]=choosen_element_uniq_val
       
        
        
        #new configurations to add to al_matrix
        new_node_to_al=matrix(rep(NA,nrow(mat)),ncol=1)
        
        
        #put in chosen element into al_matrix
        new_node_to_al[which_row_branch:(which_row_branch+choosen_element_uniq_val_nr-1),]=which_element_choosen
        
        #concatinate these vectors with mat and al_matrix
        mat=cbind(mat,new_node_value)
        al_matrix=cbind(al_matrix,new_node_to_al)
        
       
        #change scores of mat
        mat[which_row_branch:(which_row_branch+choosen_element_uniq_val_nr-1),1]=scores
        
        
        
        
      }
    } 
  }
  
  #whileloop stopped after root was added to tree redefine sco from 0 to the sum of the branch scores in mat
  if(sco==0){
    sco=sum(as.numeric(mat[,1]))
  }
 
  #return log-marginal likelihood
  return(sco)
}



############################################################################################################
#Function:func
############################################################################################################
#Input:
#data:data used.
#k:parentsize.
#node:which node to calculate log-marginal likelihood from.

#parent_comb:Specific parent combination

#fid:path of scorefile. 

#set_of_intrest:contains all node in the network .

#p:contains a list of lists where each list contains the values of a node in the network




#Output:Writes score for all combinations of set_of_intrest for all parentsets for specific parentsize.This function will be put into
#function below csitree_calc_parent_score_to_file_3.



#############################################################################################################
#import CSI algorithm
source("csi_tree_imp_2.R")
func=function(data,node,k,fid,set_of_intrest,p){
  
  
  #calculate log-marginal likelihood for empty parent
  if(k==0){ 
    N=1
    
    writeLines(paste(node, nps), con = fid, sep = "\n")
   
    
    uniq_Xi_value=matrix(p[[node]])
  
    
    nr_uniq_Xi_value1=nrow( uniq_Xi_value)
    
    
    alpha_node_parnode1=N/(nr_uniq_Xi_value1)
    alpha_node_parnode1=rep(alpha_node_parnode1,nr_uniq_Xi_value1)
    alpha_sum_parnode1=sum( alpha_node_parnode1)
    M_X_i=table(data[,..node])
    
    con=as.numeric(names(M_X_i))
    fill=rep(0,nrow(uniq_Xi_value))
    
    
    
    where_in_total=match( uniq_Xi_value,con)
    where_in_total=which(where_in_total>0)
    
    fill[where_in_total]=as.numeric(M_X_i)
    
    M_X_i=fill
    
    M_sum_count=sum(M_X_i)
   
    
    src=(lgamma(alpha_sum_parnode1)-lgamma(alpha_sum_parnode1+M_sum_count)+sum(lgamma(alpha_node_parnode1+M_X_i)-lgamma(alpha_node_parnode1)))
    
    writeLines(paste(trimws(format(round(src, 6), nsmall=6)),k, sep = " "), con = fid, sep = "\n")
    src=0
  }else{  
    
   #Else for all parentsets of node with cardinality k 
    parent_set=setdiff(set_of_intrest,node)
    parent_comb=combn(parent_set,k)
    
    #calculate all CSI-log-marginal-likelihoods
  
    lapply(1:ncol(parent_comb),function(x){ writeLines(paste(trimws(format(round(CSI_tree_apply_imp_3_mat_B_3(data,c(parent_comb[,x]),node,p),6), nsmall=6)),k,paste(parent_comb[,x],collapse = " "), sep = " "), con = fid, sep = "\n")})
    
    
  }
}



#csitree_calc_parent_score_to_file_2 is simlar to csitree_calc_parent_score_to_file_3


############################################################################################################
#Function:csitree_calc_parent_score_to_file_3
############################################################################################################
#Input:
#data:data used.
#score_type: type of score.
#File_out:Name of score file

#max_parent_size:bound on parentsize


#p:contains a list of lists where each list contains the values of a node in the network




#Output:Writes score for all combinations of set_of_intrest for all parentsets for all parentsizes up to the bound max_parent_size.



#############################################################################################################


csitree_calc_parent_score_to_file_3 <- function(data, score_type, max_parent_size, file_out,p){

  #N=1
  
  
  #number of nodes
  numcol= ncol(data) 
  #list of nodes
  set_of_intrest=1:numcol
  
  

  
 #calculate and write to file  scores of all parent combinations of all parentsets for all nodes for all parent cardinalities smaller or euqal to max_parent_size
  fid <- file(paste0("C:/Users/rasyd/Documents/gitrepo/master/score_folder/scores/csi_survey/n5000/",file_out, ".", score_type, ".score", sep = ""),"wt")
  
  writeLines(toString(numcol), con = fid, sep = "\n")
  
    #call on func and iterate over all nodes in network and all parent sizes
    
    apply(matrix(set_of_intrest,1,length(set_of_intrest)),2,function(x)apply(matrix(0:max_parent_size,1,(max_parent_size+1)),2,function(y) func(data,x,y,fid,set_of_intrest,p)))
    
    #close file
    close(fid)
  
  
 
}

















############################################################################################################
#Function:run_func_2
############################################################################################################
#Input:

#read_this:path of scorefile
#max_parent_size:how many scorefiles should be made.

#nr_nodes:how many nodes in network.

#true_matrix_2:some transformation of the adjacency matrix of the network

#j:iterator index.



#ns_i:data sample size.
#khh:which score type CSI or CPT

#Output:AUC


#This runs MCMC over a scorefile 
#############################################################################################################



func_MCMC=function(read_data,true_matrix_2,max_parent_size,j,nr_nodes,ns_i,khh){
  #change .score file to .txt file
  func_score=function(read_data){
    library("MASS")
    add=paste0("Score_csi_",toString(j),".txt")
    
    write.matrix(read_data,sep=" , ",file=add)
    
    
    score_csi_2=read.csv(file = add, header = FALSE,sep=" " )
    
    return(score_csi_2)
  }
  
  #run func_score on score-file 
  score_csi_2=(func_score(read_data))
  # remove 2 first rows containing only NA values
  score_csi_2=score_csi_2[-c(1,2),]
#list of nodes
  nodes=1:nr_nodes
  #index of nodes
  integer=(as.numeric(score_csi_2[,1]) - abs(floor(as.numeric(score_csi_2[,1])))) == 0
  
  
  integer_true=which(integer==TRUE)
  
  #index of first occurrence of parentset with specific parentset cardinality.These indexes are the same for all nodes in list nodes
  seq_3=rep(0,max_parent_size)
  for(i in 1:max_parent_size){
    seq_3[i]=choose(nr_nodes-1,i)
  }
  
 #Function for finding score with empty parentset 
  integer_zero=function(score_csi_2,int,node){
    
    index=int[node]+1
    return(as.numeric(score_csi_2[index,1]))
  }

  
  
  #Function for finding score of  parentset of a specific node
  integer_map=function(score_csi_2,input_row,integer,nodes,seq_3,nr_nodes){
    #node is in first place in input_row
    node=input_row[1]
    
    #parentset cardinality of parentset of node is in third place in input_row
    nr_parent=input_row[3]
    
   #set_1 is a set of node from 1 to first parent in parentset excluding node
    set_1=setdiff(1:input_row[4],node)
    #set_3 is a set of node from 1 to first parent in parentset 
    set_3=1:input_row[4]
    #set_2 is a set of node from  first parent to last parent excluding node 
    set_2=setdiff(input_row[4]:nodes[length(nodes)],node)
    
    
    #step will be used as the number of indexes that has to be jumped over
    step=0
    
    #Count for while loop
    n=1
    #count will be used to denote how many times one is in the for loop after n>=2
    count=1
    #number of parents excluding last parent
    node_1_len=nr_parent-1
    #number of nodes excluding node
    nr_nodes_1=nr_nodes-1
   #while true
    while(n!=0){
      
      #if n=number of parents
      if(n==length(input_row[4:(length(input_row))])){
        
        break
      }
      
      #if n is 1 
      if(n==1){
        #if node is between 1 to first parent
        if(node%in%set_3){
          #use set_1 
          set_4=set_1
        
          
        }else{
          #else use set_3
          set_4=set_3
        }
      }else{
        
        # for n>1 use set_2
      set_4=set_2
      
      }
      #if n>=2 reduce set_4 by indexes 1 to how many times in forloop after n>=2
      if(n>=2){
        
        set_4=set_4[-c(1:(count))]
      }
      
    
      #count for how many times in forloop
      count_4=0
      
      #for j from 1 to length of set_4
      for(j in 1:(length(set_4))){
        #if n>=2 find number of elements to exclude from first element in constant set set_2
        if(n>=2){
          count=count+1
        }
        
        #number of times in forloop
        count_4=count_4+1
       
        #if element j in set_4 equal parentset element n-1 in input_row
        if(set_4[j]==input_row[(4+n-1)]){
          
          #subtract number of parent with how many times inside forloop before the if statement is activated denoted by count_4
          nr_nodes_1= nr_nodes_1-count_4
          #break forloop
          break
        }
        #Add how many rows to jump over in scorefile
        step=step+choose((nr_nodes_1-j),node_1_len)
        
        
      }
      #subtract element from number of elements
      node_1_len=node_1_len-1
     
      # update n
      n=n+1
      
    }
    
    #find last parent in parentset.If parentset contains one element then the whileloop will not be used  else last 
    #parent will be between next to last parent+1 and nr_nodes
    if(nr_parent==1){
     
      #define seq_4
      seq_4=setdiff((1:nr_nodes),node)
      
    }else{
      #else we have to find the last element based on next to last element
      what=input_row[(length(input_row)-1)]
      seq_4=setdiff((what+1):nr_nodes,node)
     
    }
    #find index of last parent in parentset
    add=which(seq_4==input_row[length(input_row)])
    #add to step 
    step=step+add-1
   
    #calculate index of node parent combination
    extract_row=integer[node]+sum(seq_3[1:(nr_parent-1)])+step+2
    if(nr_parent==1){
      extract_row=integer[node]+step+2
    }
    
    #return score of node parent combination
    return(as.numeric(score_csi_2[extract_row,1]))
  }
  
  
  
  
  
  #Function for calculating neighbourhood of a DAG
  func_nabour_imp_large_B_2=function(adj){
   
    #index how many potential adds
    index_add=which(adj==0,arr.ind = TRUE)
    # how many 1's in every column
    nr_of_par_each=apply(adj,2,sum)
    
    #remove edge from node to itself
    index_add=index_add[index_add[,1] !=index_add[,2],]
    
    #number of deletes
    index_delete_rev_1=which(adj==1,arr.ind = TRUE)
    
    #number potential reverse
    index_delete_rev_2=which(adj==1,arr.ind = TRUE)
    
    #if any column has more then max_parent_size 1's 
    if(any(nr_of_par_each>=(max_parent_size))){
      #find which column
      which_add_greater_then=which(nr_of_par_each>=(max_parent_size))
      #match these elements with column vector in index_add matrix
      match_val=match(index_add[,2],which_add_greater_then)
      #remove NA
      match_val=which(match_val>0)
     #delete indexes corresponding to element in second column in index_add being equal to match_val
      if(length(match_val)!=0){
        index_add=index_add[-match_val,]
      }
    }
    
    
   #If potential reverse index matrix is none empty
    if(length(index_delete_rev_2 )!=0){
      
      # column 1 represents rows in adjecency matrix.If a reverse happens these elements will represent columns in the adjecency matrix
      current_switch=index_delete_rev_2[,1]
      # sort first column in reverse index matrix
      which_true=sort(unique(current_switch))
      
      #how many 1's exist in the columns in the adjecency matrix that will get an extra 1 by reversing
      larger_then_max=apply(matrix(adj[,c(which_true)],ncol=length(which_true)),2,function(x)sum(x))
      
      
     
      
      
      #if any of these columns already have more then max_parent_size 1's
      if(any( larger_then_max>=(max_parent_size))){
        #which index in which_true has this characteristic
        which_reverse_greater_then=which(larger_then_max>=max_parent_size)
        # find which column in which_true
        which_true= which_true[which_reverse_greater_then]
        what_to=match(1:nr_nodes,which_true)
        mmacth=which(what_to>0)
        
        match_val=match(index_delete_rev_2[,1],mmacth )
       
        match_val=which(match_val>0)
        
        
        #exclude the adjacency matrix indexes contained in first column in index_delete_rev_2 that matches match_val  
        if(length(match_val)!=0){
          index_delete_rev_2=index_delete_rev_2[-match_val,]
        }
        
      }
    }
   
    #Set bounds for number of add,delete and reverse 
    
    if(length(index_add)==0){
      index_add=c(1,2)
      
      u=0
    }else{
      u=nrow(index_add)
    }
    if(length(index_delete_rev_1)==0){
      index_delete_rev_1=c(1,2)
      l=0
    }else{
      l=nrow(index_delete_rev_1)
    }
    
    
    if(length(index_delete_rev_2)==0){
      index_delete_rev_2=c(1,2)
      pp=0
    }else{
      pp=nrow(index_delete_rev_2)
    }
    
    #define a zero matrix
    zero_matrix=array(0,c(nrow(adj),ncol(adj)))
    #Define a two dimentional storing matrix 
    B_matrix=array(0,c(nrow(adj)*(u+l+pp),ncol=ncol(adj)))
  
  #every k:m row is a matrix in B_matrix  
    k=1
    m=ncol(adj)
   
  #for j=1 add, j=2 delete and j=3 reverse
    for(j in 1:3){
      if(j==1){
        #set bound for number add checks
        f=u
      }
      
      if(j==2){
        #set bound for number delete checks
        f=l
      }
      if(j==3){
        
        #set bound for number reverse checks
        f=pp
      }
      for(i in 1:f){
        
        #if add
        if(j==1){
          
          if(is.null(nrow(index_add))){
            
            next
          }
          
          #set 1 on index index_add[i,] in adj(adjacency matrix)
          zero_matrix=adj
          zero_matrix[array(index_add[i,],c(1,2))]=1
          
          
        }
       
        #if delete
        if(j==2){
          if(is.null(nrow(index_delete_rev_1))){
           
            next
          }
          #set 0 on index index_add[i,] in adj(adjacency matrix)
          zero_matrix=adj
          zero_matrix[array(index_delete_rev_1[i,],c(1,2))]=0
          
          
          
        }
        #if reverse
        if(j==3){
          if(is.null(nrow(index_delete_rev_2))){
            
            next
          }
          #set reverse index_add[i,] in adj(adjacency matrix)
          zero_matrix=adj
          rev=array(index_delete_rev_2[i,],c(1,2))
          zero_matrix=reverse_oper_adjacent(zero_matrix,rev[1],rev[2])
          
        }
        #check whether the change made adj cyclic
        ind=is_dag(zero_matrix)
        
        #if not
        if(ind==TRUE){
          
        #if add   
          if(j==1){
            
            #add to B_matrix
            B_matrix[k:m,1:ncol(adj)]=zero_matrix
            #change k and m
            k=k+ncol(adj)
            m=k+ncol(adj)-1
          }
          #if delete
          if(j==2){
            #add to B_matrix
            B_matrix[k:m,1:ncol(adj)]=zero_matrix
            
            
            #change k and m
            k=k+ncol(adj)
            m=k+ncol(adj)-1
            
            
            
          }
          
          #if reverse
          if(j==3){
            #add to B_matrix
            B_matrix[k:m,1:ncol(adj)]=zero_matrix
            
            #change k and m
            k=k+ncol(adj)
            m=k+ncol(adj)-1
            
            
            
          }
          
        }
       
        
      }
      
      
    }
    
    
   #remove zero rows in last portion of B_matrix
    B_matrix=B_matrix[1:(k-1),]
    
    
    return(B_matrix)
  }
  
  
  

  #Function for reversing edge of adj
  reverse_oper_adjacent=function(adj,i,node){
    #make copy of adj
    adj_copy=adj
    #set value of  matrix on index node,i to value of matrix on index i,node
    adj[node,i]= adj[i,node]
    
    #set value of  matrix on index i,node to value of matrix on index node,i
    adj[i,node]=adj_copy[node,i]
    
    return(adj)
  }
  
  
  
  #Function for indexing B_matrix in neighbourhood function into matrices of adj size  
  sec_func=function(stack_matrix,adj_row){
    k=1
    m=adj_row
    nr_matrix=nrow(stack_matrix)/adj_row
   
    seq_mat=array(0,c(nr_matrix,2))
    seq_mat[1,]=c(k,m)
    if(nr_matrix>1){
      for(i in 2:nr_matrix){  
        
        k=k+adj_row
        m=k+adj_row-1
        seq_mat[i,1]=k
        
        seq_mat[i,2]=m
      }
    }
    
    return(seq_mat)
  }
  
  
  #Function returning columns where adj_1 is different from adj_2 together with the columns of both adj_1 and adj_2 where they differ
  diff_adj=function(adj_1,adj_2){
    
    
    
    
    ind=which(adj_1!=adj_2,arr.ind = TRUE)
    
    
    return(list(a=ind[,2],b=adj_1[,ind[,2]],c=adj_2[,ind[,2]]))
  }
  
  #Function for converting node,parentsize, parentset represented by 0 1 vector into parentset represented with natural numbers and finding the score for this vector
  from_adjecency_row_2=function(input_diff1,input_diff2){
    
    matr_desing_node=input_diff1
    
    nr_parent=sum(input_diff2)
    
    
    parent=which(input_diff2==1)
    
    
  
    if(nr_parent>0){
      score=integer_map(score_csi_2,c( matr_desing_node,123,nr_parent,parent),integer_true,nodes,seq_3,nr_nodes)
    }else{
      score=integer_zero(score_csi_2,integer_true,matr_desing_node)
    }
    
    
    return(score)
    
  }
  
  

  
  
  #Function for converting adjacency matrix  into a set of (node,parentsize,parentset) represented with 
  #natural numbers in order to find the score for the adjacency matrix.  
  
  from_adjecency_to_matrx=function(adj){
    
     
    matr_desing=matrix(0,ncol(adj),ncol(adj)+3)
    
    matr_desing[,1]=1:ncol(adj)
    matr_desing[,2]=1
    matr_desing[,3]=apply(adj,2,sum)
    
    v=apply(adj,2,function(x) which(x==1))
    
    
    for(j in 1:nrow(matr_desing)){
      if(matr_desing[j,3]==0){
        next
      }
      
      parent=v[j][[1]]
      
      len_par=length(parent)
      
      matr_desing[j,4:(4+len_par-1)]=sort(parent)
    }
    

    for(i in 1:nrow(matr_desing)){
      if(matr_desing[i,3]==0){
        
        matr_desing[i,2]= integer_zero(score_csi_2,integer_true,matr_desing[i,1])
        
      }else{
        
        
        none_zero=match(0,matr_desing[i,])-1
        
        
    
        
        matr_desing[i,2]= integer_map(score_csi_2,matr_desing[i,1:none_zero],integer_true,nodes,seq_3,nr_nodes)
        
        
      }
    }  
    
    return(matr_desing[,2])
    
  }


  #number of iterations
  N=600000
  #thinning
  lag=10
  #how many MCMC chains
  m=1
  #start collecting samples after burn in
  start=550000
  
  #stop collecting samples
  stop=600000
  
  #sample collection sequence
  save_every=seq(start,stop,lag)
 
  #set initial DAG for chains
  inital_value_adj=matrix(0,m*nr_nodes,nr_nodes)
 
  #Slice inital_value_adj depending on m
  slice_intial=sec_func(inital_value_adj,nr_nodes)
  
  #calculate initial score
  inital_value_scores=matrix(0,nr_nodes,m)
  for(i in 1:dim(slice_intial)[1]){
    
    
    inital_value_scores[,i]=(from_adjecency_to_matrx(inital_value_adj[slice_intial[i,1]:slice_intial[i,2],]))
    
  }
  
  
  inital_value_score=colSums(inital_value_scores)
  
  #define storage matrix for traversal of chain in score space
  X_t_matrix=matrix(0,m,N)
  #put initial score as first column in storage matrix X_t_matrix
  X_t_matrix[,1]=inital_value_score
  
  
  #matrix for saving samples 
  save_array=array(0,c(nr_nodes*(length(save_every)),nr_nodes,m))
  
  #slice save matrix
  slice_save_array=sec_func(save_array,nr_nodes)

  #iterator for slice_save_array
  acc_vec=rep(0,m)
  
  
  w=0
  
  #Save currently accepted matrix in matr_current
  matr_current=array(0,c(m*nr_nodes,nr_nodes))
  #Save currently proposed matrix in matr_prop
  matr_prop=array(0,c(m*nr_nodes,nr_nodes))
  
  start=Sys.time()
  
  #for 2 to N
  for(i in 2:(N)){
    #for 1 to m chains
    for(k in 1:m ){
      
      # set initial DAG to matr_current for chain k and calculate neighbourhood of inital DAG 
      if(w==0){
        matr_current[slice_intial[k,1]:slice_intial[k,2],]=inital_value_adj[slice_intial[k,1]:slice_intial[k,2],]
        neighbour_candate=func_nabour_imp_large_B_2(matr_current[slice_intial[k,1]:slice_intial[k,2],])
        
      }
      
    #slice neighbour_candidate
      neighboors_cand_slice=sec_func(neighbour_candate,nr_nodes)
      #number neighbours in neighbour_candidate
      nr_of_neighboors=nrow(neighboors_cand_slice)
    
      #sample uniform one DAG from neighbour_candidate
      matr_prop_nr=sample.int(nrow(neighboors_cand_slice),1)
      
     #put it equal to matr_prop
      matr_prop[slice_intial[k,1]:slice_intial[k,2],]=neighbour_candate[neighboors_cand_slice[matr_prop_nr,1]:neighboors_cand_slice[matr_prop_nr,2],]
     
      #Collect matr_current at state i
      if(i%in% save_every){
        
        #go to next index in slice_save_array
        acc_vec[k] = acc_vec[k]+1
       # save matr_current in state i in position acc_vec[k],1]:slice_save_array[acc_vec[k],2],,k in save_array
        save_array[slice_save_array[acc_vec[k],1]:slice_save_array[acc_vec[k],2],,k]=matr_current[slice_intial[k,1]:slice_intial[k,2],]
        
      }
    
      #calculate neighbourhood of current state of prop_matr
      prop_neighbor=func_nabour_imp_large_B_2(matr_prop[slice_intial[k,1]:slice_intial[k,2],])
      
      #number of neighbours in prop_neighbor
      prop_neighbor_nr=nrow(sec_func(prop_neighbor,nr_nodes))
     
      #find difference between current state of matr_current and current matr_prop
      diff_c_p=diff_adj(matr_current[slice_intial[k,1]:slice_intial[k,2],],matr_prop[slice_intial[k,1]:slice_intial[k,2],])
     
      #If one column is returned it means the difference is an add or delete
      if(length(unlist(diff_c_p$a))==1){
        
        #Define which  column differ
        extract_1=as.numeric(diff_c_p$a)
        #column in matr_current
        extract_2=diff_c_p$b
        #column in matr_prop
        extract_3=diff_c_p$c
        #score of column in matr_current
        what_to_delete=from_adjecency_row_2(extract_1,extract_2)
        #score of this column in matr_prop
        what_to_add=from_adjecency_row_2(extract_1,extract_3)
        
        #compute score of matr_prop by subtracting score of column in matr_current and adding score of this column in matr_prop
        prop_score=X_t_matrix[k,(i-1)]+what_to_add-what_to_delete }else{
          
          #else the difference is an reverse if the neigbourhood function is correct.
         
          #Define which  column differ
          extract_1=t(diff_c_p$a)
          
           #column in matr_current
          extract_5=diff_c_p$b
          #column in matr_prop
          extract_6=diff_c_p$c
         
          #score of column in matr_current
          what_to_delete=apply(matrix(c(1,2),1,2),2,function(x) from_adjecency_row_2(extract_1[,x],extract_5[,x]))
          #score of this column in matr_prop
          what_to_add=apply(matrix(c(1,2),1,2),2,function(x) from_adjecency_row_2(extract_1[,x],extract_6[,x]))
          
          #compute score of matr_prop by subtracting score of column in matr_current and adding score of this column in matr_prop
          prop_score=X_t_matrix[k,(i-1)]+sum(what_to_add)-sum(what_to_delete) 
        }
      
      #calculate acceptance ratio
      R = min(1,exp((prop_score+log(1/prop_neighbor_nr))-(X_t_matrix[k,(i-1)]+log(1/nr_of_neighboors))))
      
      
      
      #Sample random uniform number and check if its smaller or equal to acceptance rate
      if(runif(1)<=R)
      {
        #if this is the case add matr_prop score to X_t_matrix
        X_t_matrix[k,(i)]=prop_score 
        #Change matr_current to matr_prop
        matr_current[slice_intial[k,1]:slice_intial[k,2],]=matr_prop[slice_intial[k,1]:slice_intial[k,2],]
        #calculate neighbourhood of matr_current
        neighbour_candate=func_nabour_imp_large_B_2(matr_current[slice_intial[k,1]:slice_intial[k,2],])
        
        
        
        #increase w
        w=w+1
        
      }else{
        #else do not change matr_current and set score t equal to score t-1
        X_t_matrix[k,(i)]=X_t_matrix[k,(i-1)]
      }
      
      
      
    }
    
    
    
  }

  
  #transform the samples matrixes where each element in the matrix represent how many ways irrelevant of path length one can go from parent(column) to node(row)
  gemetric_s_sum=lapply(1:nrow(slice_save_array),function(x)solve(diag(nr_nodes)-save_array[slice_save_array[x,1]:slice_save_array[x,2],,1]))
  
  
  #transform matrices in gemetric_s_sum into 1 0 matrices indicating if there is at least 1 way to go from parent(column) to node(row).
  one_path=lapply(1:length(gemetric_s_sum), function(x) gemetric_s_sum[[x]]>=1)
  
  #take an average of matrices in one_path.
  edge_matrix_sum=Reduce('+', one_path)
  
  
  edge_matrix=edge_matrix_sum/length(one_path)
  
  
  edge_matrix[row(edge_matrix)==col(edge_matrix)]=1
  
  
  #take an average of the samples.This average shows the estimate for direct one move edge between parent(column) to node(row).
  direct_cause=lapply(1:nrow(slice_save_array),function(x)save_array[slice_save_array[x,1]:slice_save_array[x,2],,1])
  direct_mat=Reduce('+', direct_cause)
  direct_mat=direct_mat/length(one_path)
  
  

  
  
 #this function returns random string  based on vector vec
  randstr <- function(vec) {
    characters=vec[1]
    numbers=vec[2]
    
    lowerCase=vec[3]
    upperCase=vec[4]
    ASCII <- NULL
    
    if(numbers>0)    ASCII <- c(ASCII, sample(48:57, numbers,replace = TRUE))
    if(upperCase>0)  ASCII <- c(ASCII, sample(65:90, upperCase,replace = TRUE))
    if(lowerCase>0)  ASCII <- c(ASCII, sample(97:122, lowerCase,replace = TRUE))
    if(characters>0) ASCII <- c(ASCII, sample(c(65:90, 97:122), characters,replace = TRUE))
    
    return( rawToChar(as.raw(sample(ASCII, length(ASCII)))) )
  }
  
  samp_1=sample(4:6,1,replace = TRUE)
  
  samp_2=sample(1:samp_1,4,replace = TRUE)
  
  uniq=randstr(samp_2)
  
  if(khh==1){
    keep_adding="cat"}
  if(khh==2){
    keep_adding="csi"
  }

  string_to_add=paste0("C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/csi_sachs/causal_mechanism/plot2/",toString(j),".txt")
  
  write.matrix(edge_matrix,sep=" , ",file=string_to_add)
  
  
  
  samp_1=sample(4:6,1,replace = TRUE)
  
  samp_2=sample(1:samp_1,4,replace = TRUE)
  
  uniq=randString(samp_2)
  
  if(khh==1){
    keep_adding="cat"}
  if(khh==2){
    keep_adding="csi"
  }
  
  #string_to_add=paste0("C:/Users/rasyd/Documents/gitrepo/master/score_folder/matrix/cat_child/direct_cause/n1001/",keep_adding,toString(ns_i),uniq,toSting(j),".txt")
  #toSting(j+1)
  #write.matrix(direct_mat,sep=" , ",file=string_to_add)
  
  
  
  
  
  
# Function for sorting estimate probabilities and true ancestral relationship matrix in same order and concatinating them   
  class_prob=function(compare){
    
    
    rechape_compare=c(compare)
    rechape_compare_order=order(rechape_compare,decreasing = TRUE)
    
    order_MCMC=rechape_compare[rechape_compare_order]
    
    
    rechape_true=c(true_matrix_2)
    order_true=rechape_true[rechape_compare_order]
    
    
    compare_true_with_mcmc=cbind(order_MCMC,order_true)
    
   
    
    
    return(compare_true_with_mcmc)
    
  }
  r=c("edge_matrix")
  
  
  
  #calculate AUC
  AUC_calc=AUC(class_prob(get(r))[,1],class_prob(get(r))[,2])
  
  #return AUC
  return(AUC_calc)
  
}











