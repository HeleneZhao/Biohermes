###############################
## Biohermes
## 2024-7-15
## Hanjun Zhao
###############################
#R version 4.5.1
#install.packages(c("dplyr", "ggplot2"))

## loading library 
library(verification)
library(survival)
library(timeROC)
library(rms)
library(pROC)
library(ggrepel)
library(ggVennDiagram)
library(purrr)
library(sva)
library(NMF)
library(readr) 
library(tidyverse)
library(dplyr)
library(data.table)
library(readxl)
library(FactoMineR)
library(factoextra)
library(ggpubr)
library(grid)
library(gridExtra)
library(RColorBrewer)
library(limma)
library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
library(ggstatsplot)
library(GSEABase)
library(GSVA)
library(pheatmap)
library(matrixStats)
library(ggVennDiagram)
library(EnhancedVolcano)
library(ggplot2)
library(ggtext)

##
rm(list=ls())
load('./Biohermes.RData')


datTraits0=datTraits0[datTraits0$RowCheck=="PASS",]
datexp=datexp[rownames(datexp) %in% datTraits0$USUBJID,]


table(rownames(datexp)==datTraits0$USUBJID)
gap_cv=cbind(datTraits0,datexp)

############IQR based outlier expression level detections############
#insprired by https://www.nature.com/articles/s41467-025-62463-w#Sec9

protein_7K_mat <- datexp

##samples are rows and proteins are columns
IQR_values <- apply(protein_7K_mat, 2, function(x) IQR(x, na.rm = TRUE))
Q1_values <- apply(protein_7K_mat, 2, function(x) quantile(x, probs = 0.25, na.rm = TRUE))
Q3_values <- apply(protein_7K_mat, 2, function(x) quantile(x, probs = 0.75, na.rm = TRUE))


##Identify outliers and replace with NAs
for (i in 1:ncol(protein_7K_mat)) {
  print(i)
  lower_bound <- Q1_values[i] - 2.25 * IQR_values[i]
  upper_bound <- Q3_values[i] + 2.25 * IQR_values[i]
  outliers <- which((protein_7K_mat[,i] < lower_bound) | (protein_7K_mat[,i] > upper_bound))
  protein_7K_mat[outliers,i] <- NA
}
protein_7K_mat <- as.matrix(protein_7K_mat)
protein_7K_mat[is.nan(protein_7K_mat)] <- NA
sum(is.na(protein_7K_mat))

################### 65% call rate ################
#Remove Analytes and Samples with <65% call rate
all_analyte<- colnames(protein_7K_mat)
all_subject<- rownames(protein_7K_mat)

##calculate call rate for analytes
call_rate_per_analyte<- data.frame()

for (i in (1:ncol(protein_7K_mat))){
  print(i)
  outlier_subject_number<- sum(is.na(protein_7K_mat[,i]))
  call_rate_per_analyte[i,1] <- all_analyte[i]
  call_rate_per_analyte[i,2]<- (nrow(protein_7K_mat)-outlier_subject_number)/nrow(protein_7K_mat)*100
}

colnames(call_rate_per_analyte)<- c("Analyte","Call Rate")


##calculate call rate for samples
call_rate_per_subject<- data.frame()

for (i in (1:nrow(protein_7K_mat))){
  print(i)
  outlier_analyte_number<- sum(is.na(protein_7K_mat[i,]))
  call_rate_per_subject[i,1] <- all_subject[i]
  call_rate_per_subject[i,2]<- (ncol(protein_7K_mat)-outlier_analyte_number)/ncol(protein_7K_mat)*100
}

colnames(call_rate_per_subject)<- c("Subject","Call Rate")

sum(call_rate_per_analyte$`Call Rate` < 65) 
sum(call_rate_per_subject$`Call Rate` < 65) 


###check and keep subjects with call rate more than 65%
pass_samples <-  call_rate_per_subject$Subject[call_rate_per_subject$`Call Rate` >= 65]
length(pass_samples) 

##list of 2 subjects failed 
failed_subjects_with_call_rate <- call_rate_per_subject[call_rate_per_subject$`Call Rate` < 65, ]
length(failed_subjects_with_call_rate$Subject) 

###check and keep analytes with call rate more than 65% 
pass_analytes <-  call_rate_per_analyte$Analyte[call_rate_per_analyte$`Call Rate` >= 65]
length(pass_analytes) 

failed_analytes_with_call_rate <- call_rate_per_analyte[call_rate_per_analyte$`Call Rate` < 65, ]
length(failed_analytes_with_call_rate$Analyte) 

##remove failed subjects
protein_7K_mat_post_65_check <- protein_7K_mat[pass_samples, pass_analytes]
dim(protein_7K_mat_post_65_check)


########85% call rate#########
#####Re-calculate call rate for analytes and subject, and remove them with call rate <85%#####

##extract all analyte and subjects
all_analyte<- colnames(protein_7K_mat_post_65_check)
all_subject<- rownames(protein_7K_mat_post_65_check)

##calculate call rate for analytes
call_rate_per_analyte<- data.frame()

for (i in (1:ncol(protein_7K_mat_post_65_check))){
  print(i)
  outlier_subject_number<- sum(is.na(protein_7K_mat_post_65_check[,i]))
  call_rate_per_analyte[i,1] <- all_analyte[i]
  call_rate_per_analyte[i,2]<- (nrow(protein_7K_mat_post_65_check)-outlier_subject_number)/nrow(protein_7K_mat_post_65_check)*100
}

colnames(call_rate_per_analyte)<- c("Analyte", "Call Rate")

###only keep analytes with call rate more than 85%
sum(call_rate_per_analyte$`Call Rate` < 85) 
pass_analytes <- call_rate_per_analyte$Analyte[call_rate_per_analyte$`Call Rate` >= 85]
length(pass_analytes) # 7546


##check which analytes failed
failed_analytes_with_call_rate <- call_rate_per_analyte[call_rate_per_analyte$`Call Rate` < 85, ]

####re calculate sample wise call rate####
call_rate_per_subject<- data.frame()

for (i in (1:nrow(protein_7K_mat_post_65_check))){
  print(i)
  outlier_analyte_number<- sum(is.na(protein_7K_mat_post_65_check[i,]))
  call_rate_per_subject[i,1] <- all_subject[i]
  call_rate_per_subject[i,2]<- (ncol(protein_7K_mat_post_65_check)-outlier_analyte_number)/ncol(protein_7K_mat_post_65_check)*100
}

colnames(call_rate_per_subject)<- c("Subject", "Call Rate")

sum(call_rate_per_subject$`Call Rate` < 85) ## 3
pass_samples <- call_rate_per_subject$Subject[call_rate_per_subject$`Call Rate` >= 85]
length(pass_samples) 


failed_subjects_with_call_rate <- call_rate_per_subject[call_rate_per_subject$`Call Rate` < 85, ]
##remove failed subjects
length(pass_samples) 
length(pass_analytes) 
matrix_post_85prcnt_check <- protein_7K_mat_post_65_check[pass_samples, pass_analytes]
dim(matrix_post_85prcnt_check)



############# Final Check ##########
###check final call rate and back transform to save

##calculate call rate for analytes
call_rate_per_analyte<- data.frame()

for (i in (1:ncol(matrix_post_85prcnt_check))){
  outlier_subject_number<- sum(is.na(matrix_post_85prcnt_check[,i]))
  call_rate_per_analyte[i,1] <- all_analyte[i]
  call_rate_per_analyte[i,2]<- (nrow(matrix_post_85prcnt_check)-outlier_subject_number)/nrow(matrix_post_85prcnt_check)*100
}

colnames(call_rate_per_analyte)<- c("Analyte","Call Rate")

sum(call_rate_per_analyte$`Call Rate` < 85)


call_rate_per_subject<- data.frame()

for (i in (1:nrow(matrix_post_85prcnt_check))){
  print(i)
  outlier_analyte_number<- sum(is.na(matrix_post_85prcnt_check[i,]))
  call_rate_per_subject[i,1] <- all_subject[i]
  call_rate_per_subject[i,2]<- (ncol(matrix_post_85prcnt_check)-outlier_analyte_number)/ncol(matrix_post_85prcnt_check)*100
}

colnames(call_rate_per_subject)<- c("Subject","Call Rate")

sum(call_rate_per_subject$`Call Rate` < 85)


#########################################Final matrix ########################################################
##also to remove two analytes as part of QC
Annotation_Info <- annot1 %>% filter(Protein %in% names(as.data.frame(matrix_post_85prcnt_check)))

table(Annotation_Info$Organism)
Annotation_Info<- Annotation_Info[Annotation_Info$Organism=="Human" | Annotation_Info$Organism %in% c("HIV-1","HIV-2"),]


table(Annotation_Info$Type)
Annotation_Info<- Annotation_Info[Annotation_Info$Type =="Protein",]

##subset prot matrix
matrix_post_85prcnt_check <- as.data.frame(matrix_post_85prcnt_check) %>% dplyr::select(all_of(Annotation_Info$Protein))
dim(matrix_post_85prcnt_check)


# re load raw data and log10
protein_7K_mat <- datexp
final_matrix=protein_7K_mat[rownames(protein_7K_mat) %in% rownames(matrix_post_85prcnt_check), colnames(protein_7K_mat) %in% colnames(matrix_post_85prcnt_check)]
dim(final_matrix) 

sum(is.nan(as.matrix(final_matrix)))
sum(is.na(as.matrix(final_matrix)))

write.csv(final_matrix,"./temp_result/QCpass_Plasma7K_post_85prcnt_check.csv",row.names = TRUE)
save(final_matrix, file="./temp_result/QCpass_Plasma7K_post_85prcnt_check.RData")

# rebind expr and trait matrix
final_matrix=cbind(rownames(final_matrix),final_matrix)
gap_cv=inner_join(datTraits0, final_matrix, by=c("USUBJID"="rownames(final_matrix)"))
dim(gap_cv)


#### Supplementary codes to generate the pca modified protein matrix ####
table(is.na(gap_cv$AGE))
table(is.na(gap_cv$SEX))
table(gap_cv$RACE)
table(gap_cv$ETHNIC)
table(gap_cv$SITEID)

gap_cv$SEXmod <- ifelse(gap_cv$SEX == "F", 1L, 0L) #(typically Male = 0, Female = 1)
gap_cv$AGE=as.numeric(gap_cv$AGE)

table(gap_cv$QVAL)
table(gap_cv$diagnosis)

#### Detect Flag Sample
# Flag sample

## alternative covariants
table(gap_cv$APOE)
table(gap_cv$APOEmod) #602 359

cross_apoe_diag=as.data.frame(table(gap_cv$QVAL,gap_cv$APOEmod))
colnames(cross_apoe_diag)=c('diagnosis','apoemod','freq')

gap_cv$APOEmod=as.numeric(gap_cv$APOEmod)
gap_cv$NVSTRESC_SUVR=as.numeric(gap_cv$NVSTRESC_SUVR)


table(gap_cv$DGstatus)
table(is.na(gap_cv$DGstatus))

table(gap_cv$DGstatus2)
table(is.na(gap_cv$DGstatus2))

gap_cv <- gap_cv %>%
  mutate(DGstatus3 = case_when(
    diagnosis == 'CN' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "CN-",
    diagnosis == 'CN' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "CN+",
    diagnosis == 'MCI' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "CI-",
    diagnosis == 'MCI' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "CI+",
    diagnosis == 'AD' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "CI-",
    diagnosis == 'AD' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "CI+",
    # Catch any cases that don't match, leaving them as NA
  ))
table(gap_cv$DGstatus3)



gap_cv <- gap_cv %>%
  mutate(DGstatus4 = case_when(
    diagnosis == 'CN' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "CN_A-",
    diagnosis == 'CN' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "A+",
    diagnosis == 'MCI' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "SNAP",
    diagnosis == 'MCI' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "A+",
    diagnosis == 'AD' & NVSTRESC_AMYCLAS == 'NEGATIVE' ~ "SNAP",
    diagnosis == 'AD' & NVSTRESC_AMYCLAS == 'POSITIVE' ~ "A+",
    # Catch any cases that don't match, leaving them as NA
  ))
table(gap_cv$DGstatus4)
table(is.na(gap_cv$DGstatus4)) 

table(gap_cv$apet_mod)
table(is.na(gap_cv$apet_mod))
table(gap_cv$diagmod)
gap_cv$diagmod=as.numeric(gap_cv$diagmod)

################################################################################
################################################################################
# cutting outliers
# identifying the features
variable.names=colnames(gap_cv)[211:7498]
gap_cv[,variable.names] <- apply(gap_cv[,variable.names],2,as.numeric)

# identifying the samples
length(unique(gap_cv$USUBJID))

range(gap_cv[211:7498]) 
range(rowVars(as.matrix(gap_cv[211:7498]))) 
range(colVars(as.matrix(gap_cv[211:7498])))


## PCA
pca <- prcomp(gap_cv[211:7498], scale = T) #log10 samples, scale


####
## check PCA
pcasd=pca$sdev^2/sum(pca$sdev^2)

## select the PCA as covariants
pcadf=as.data.frame(-1*pca$x)
gap_cv[,c(209:210)]=as.data.frame(pcadf[c(1:2)])


################################################################################

gap_cv[,c(211:7498)] <- scale(gap_cv[,c(211:7498)])
variable.names=colnames(gap_cv)[211:7498]
gap_cv[,variable.names] <- apply(gap_cv[,variable.names],2,as.numeric)


Uni_glm_model1 <- function(x) {
  ML=as.formula(paste0("diag.mod","~", x,"+ AGE + SEXmod + PC1 +PC2")) ## proteomics must be numeric
  Proname=x
  glm1=glm(formula=ML, family=binomial, data=df)
  LSUM=summary(glm1)
  beta=LSUM$coefficients[-1, "Estimate"]
  pvalue=LSUM$coefficients[-1, "Pr(>|z|)"]
  Uni_glm=cbind(Proname, beta, pvalue)
  Uni_glm=as.data.frame(Uni_glm)
  dimnames(Uni_glm)[[2]]=c("Protein", "AD_estimate", "pvalue")
  return(Uni_glm)
}

Uni_glm_model2 <- function(x) {
  ML=as.formula(paste0("diag.mod","~", x,"+ AGE + SEXmod + APOEmod + PC1 + PC2"))
  Proname=x
  glm1=glm(formula=ML, family=binomial, data=df)
  LSUM=summary(glm1)
  beta=LSUM$coefficients[-1, "Estimate"]
  pvalue=LSUM$coefficients[-1, "Pr(>|z|)"]
  Uni_glm=cbind(Proname, beta, pvalue)
  Uni_glm=as.data.frame(Uni_glm)
  dimnames(Uni_glm)[[2]]=c("Protein", "AD_estimate", "pvalue")
  return(Uni_glm)
}


################################################################################
## Differential analysis for AD and CN

for(j in c("AD")){
  df=filter(s.data, s.data$diagnosis %in% c(j,"CN"))
  df$diag.num=ifelse(df$diagnosis=="CN","0","1")
  df$diag.num=as.numeric(df$diag.num)
  df$diag.mod=as.factor(df$diag.num)
  Uni_glm_result=vector(mode="list",length=length(variable.names))
  Uni_glm_result=lapply(variable.names,Uni_glm_model1)
  result=data.frame()
  Uni_glm2=vector(mode="list",length=length(variable.names))
  for (i in 1:length(variable.names)){
    Uni_glm2[[i]]=Uni_glm_result[[i]][-which(row.names(Uni_glm_result[[i]])%in%c("AGE","SEXmod","PC1","PC2")),]
    result=rbind(result,Uni_glm2[[i]])
  }
  result$pvalue=as.numeric(result$pvalue)
  result=result[order(result$pvalue),]
  result$padjust_BH=p.adjust(result$pvalue, method="BH")
  ## Annotation
  result_ann=left_join(result,annot1,by=c("Protein"="Protein"))
  ## Saving result
  filename=paste0("./temp_result/1.LG_AD_vs_CN_agesex_s/1.LR_",j,"_vs_CN_agesex.csv")
  write.csv(result_ann,filename,row.names=FALSE)
}


#######################################################################################

#### 1.Data input, cleaning and pre-processing
#### 1.a Loading expression data ####


datTraits0=datTraits0[datTraits0$DGstatus %in% c("AD_A+","MCI_A+","CN_A+","CN_A-"),]
datexp = datexp[row.names(datexp) %in% datTraits0$USUBJID,]
pca <- prcomp(datexp,scale = T)
pcadf=as.data.frame(-1*pca$x)

## Check variance
range(datexp)
regress_out <- function(datexp, covariates) {
  (apply(datexp, 2, function(protein) { 
    model_data <- data.frame(Expression = protein, covariates)  
    model <- lm(Expression ~ ., data = model_data) 
    residuals(model)+mean(protein)
  }))
}
covariates <- gap_cv[, c("AGE","SEX","PC1","PC2")]  # Ensure these column names exist

datexp_adjusted <- regress_out(datexp, covariates)
range(datexp_adjusted) 


table(rownames(datexp_adjusted)==datTraits0$USUBJID)
gap_cv_reg=cbind(datTraits0, datexp_adjusted)


##################################
#### 1.b Checking data for excessive missing values and identification of outlier samples####
gsg = goodSamplesGenes(datexp_adjusted, verbose = 3)
gsg$allOK #TRUE
if (!gsg$allOK)
{
  # Optionally, print the gene and sample names that were removed:
  if (sum(!gsg$goodGenes)>0)
    printFlush(paste("Removing genes:", paste(names(datexp_adjusted)[!gsg$goodGenes], collapse = ", ")));
  if (sum(!gsg$goodSamples)>0)
    printFlush(paste("Removing samples:", paste(rownames(datexp_adjusted)[!gsg$goodSamples], collapse = ", ")));
  # Remove the offending genes and samples from the data:
  datexp_adjusted= datexp_adjusted[gsg$goodSamples, gsg$goodGenes]
}

## Clustering samples to see if there is any outliers
sampleTree = hclust(dist(datexp_adjusted), method = "average");

clust = cutreeStatic(sampleTree, cutHeight = 22, minSize = 10) 
table(clust)
datexp_adjusted = datexp_adjusted[clust == 1, ]
datTraits_lab = datTraits0[(datTraits0$USUBJID %in% row.names(datexp_adjusted)),]
table(datTraits_lab$USUBJID==row.names(datexp_adjusted))
dim(datexp_adjusted) 
nGenes = ncol(datexp_adjusted) 
nSamples = nrow(datexp_adjusted)

traitColors = numbers2colors(datTraits_lab[,c(203,211)], signed = FALSE)
sampleTree = hclust(dist(datexp_adjusted), method = "average");


#########################################################################################
powers = c(seq(from = 1, to=6, by=0.5), seq(from = 7, to=9, by=1), seq(from = 10, to=30, by=2))

## Call the network topology analysis function
sft = pickSoftThreshold(datexp_adjusted,
                        networkType = "signed",
                        powerVector = powers,
                        verbose = 5)
sft$powerEstimate 

sizeGrWindow(9,5)
par(mfrow = c(1,2));
cex1 = 0.9;

## Scale-free topology fit index as a function of the soft-thresholding power
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"));
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers, cex=cex1,col="red");
## this line corresponds to using an R^2 cut-off of h
abline(h=cex1, col="red")

## Mean connectivity as a function of the soft-thresholding power
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")

net_fine <- blockwiseModules(
  datexp_adjusted,            
  power              = 6,
  networkType        = "signed",  
  TOMType            = "signed",
  corType            = "bicor",    
  deepSplit          = 4,           
  minModuleSize      = 15, 
  reassignThreshold  = 0,         
  pamStage           = T,    
  pamRespectsDendro  = T,      
  mergeCutHeight     = 0.25,      
  numericLabels      = TRUE,
  saveTOMs           = FALSE,
  verbose            = 3
)
table(net_fine$colors)
sizeGrWindow(12, 9)

moduleColors = labels2colors(net_fine$colors)
table(moduleColors)

png(file = "./temp_result/5.WGCNA/5.WGCNA_Dendrogram_and_modulecolors.png", width = 3000, height = 2000, res = 300)

## Plot the dendrogram and the module colors underneath
plotDendroAndColors(net_fine$dendrograms[[1]], 
                    moduleColors[net_fine$blockGenes[[1]]],
                    "Module colors",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05)
dev.off()


########################################################################################
nGenes = ncol(datexp_adjusted)
nSamples = nrow(datexp_adjusted)

## Recalculate MEs with color labels
MEs0 = moduleEigengenes(datexp_adjusted, moduleColors)$eigengenes
MEs = orderMEs(MEs0)


##################################################################################
moduleTraitCor = cor(MEs, datTraits_lab[,c(151,139,142,140,145,149:150,155,175,70,125)], use = "p");
moduleTraitPvalue = corPvalueStudent(moduleTraitCor, nSamples);

sizeGrWindow(15,6)
textMatrix = paste(signif(moduleTraitCor, 2), "\n(",
                   signif(moduleTraitPvalue, 1), ")", sep = "");
dim(textMatrix) = dim(moduleTraitCor)

png(file = "./temp_result/5.WGCNA/5.WGCNA_Heatmap_modulecolors.png", width = 8000, height = 6000, res = 600)
par(mar = c(4, 6, 4, 0));

labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = colnames(datTraits_lab[,c(151,139,142,140,145,149:150,155,175,70,125)]),
               yLabels = names(MEs),
               ySymbols = names(MEs),
               colorLabels = FALSE,
               colors = greenWhiteRed(50),
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 0.5,
               zlim = c(-1,1),
               main = paste("Module-trait relationships"))
dev.off()

modNames = substring(names(MEs), 3)
modNames


## Kruskal-Wallis test
# Prepare data
MEs <- MEs %>%
  as.data.frame() %>%
  mutate(USUBJID = row.names(.)) %>%
  relocate(USUBJID)

# Merge datasets
mes_group <- inner_join(MEs, datTraits_lab, by = "USUBJID")
results_list <- list()

group_col <- "DGstatus"
desired_order <- c("CN_A-", "CN_A+", "MCI_A+", "AD_A+")
mes_group[[group_col]] <- factor(mes_group[[group_col]], levels = desired_order)

kw_results_list <- list()
kw_tbl <- do.call(rbind, lapply(names(MEs), function(var){
  p <- kruskal.test(as.formula(paste(var, "~", group_col)), data = mes_group)$p.value
  data.frame(module = var, p = p)
}))
kw_tbl$q <- p.adjust(kw_tbl$p, method = "BH")

#############################


                   





