#### MR for Bio_hermes ####

## library
library(TwoSampleMR)
library(ggplot2)
library(foreach)
library(genetics.binaRies)
library(ieugwasr)
library(R.utils)
library(coloc)
library(dplyr)
library(readr)
library(tidyverse)
library(data.table)
library(readxl)
library(stringr)
library(ggplot2)
library(RColorBrewer)
library(nortest)
library(ggpubr)
library(sva)
library(matrixStats)
library(dbscan)
library(plyr)
library(broom)
library(limma)
library(caret)
library(nortest)
library(ggpubr)
library(sva)
library(matrixStats)
library(MASS)
library(tidyr)
library(lmerTest)
library(statar)
library(R.utils)
library(readxl)
library(devtools)
library(org.Hs.eg.db)
library(clusterProfiler)
library(SNPlocs.Hsapiens.dbSNP155.GRCh38)
library(MungeSumstats)
library(plinkbinr)
library(GenomicRanges)
library(BiocParallel)


protein_List <- list.files(
  path = "/home/MR_rawdata/Bioherme_DEP_GWAS/GWAS_A_DAP/",
  pattern = "^gwas_seq\\..*\\.assoc\\.linear$",
  full.names = TRUE
)

gwas_annt <- fread("/home/MR_rawdata/Bioherme_DEP_GWAS/gwas_with_rsid.tsv")
fred_file <- fread("/home/MR_rawdata/Bioherme_DEP_GWAS/GWAS_A_DAP/freq.frq")
snp_meta <- inner_join(gwas_annt, fred_file, by=c("SNPID"="SNP"))


## MR

mr_res <- data.frame()
mr_resr <- data.frame()
pval_threshold <- 5e-06 
eaf_threshold <- 0.05
clump_kb_val <- 1000
clump_r2_val <- 0.01
cis_window <- 1000000

is_palindromic_vec <- function(a1, a2){
  a1 <- toupper(a1); a2 <- toupper(a2)
  (a1 == "A" & a2 == "T") | (a1 == "T" & a2 == "A") |
    (a1 == "C" & a2 == "G") | (a1 == "G" & a2 == "C")
}

unprocessed_proteins <- data.frame()
unprocessed_proteins_file <- paste0("/home/MR_rawdata/Bioherme_DEP_GWAS/MR_results/proteins_unprocessed.txt")
annot=read.csv("/home/MR_rawdata/Bioherme_DEP_GWAS/protein_annotation.csv")

is_palindromic_vec <- function(a1, a2){
  a1 <- toupper(a1); a2 <- toupper(a2)
  (a1 == "A" & a2 == "T") | (a1 == "T" & a2 == "A") |
    (a1 == "C" & a2 == "G") | (a1 == "G" & a2 == "C")
}

# https://www.ebi.ac.uk/gwas/studies/GCST90027158
out_gwas <- as.data.frame(read_tsv("/home/MR_rawdata/dementia_data/GCST90027158_buildGRCh38.tsv"))

out_gwas$phenotype <- "AD"
out_gwas$samplesize <- out_gwas$n_cases + out_gwas$n_controls

outcome_format <-  format_data(
  out_gwas,
  type = "outcome",
  snp_col='variant_id',
  chr_col = "chromosome",
  pos_col = "base_pair_location",
  beta_col = 'beta',
  se_col = 'standard_error',
  effect_allele_col = 'effect_allele',
  other_allele_col = 'other_allele',
  eaf_col = 'effect_allele_frequency',
  pval_col = 'p_value',
  ncase_col = "n_cases",
  ncontrol_col = "n_controls",
  samplesize_col = 'samplesize',
  phenotype_col = 'phenotype'
)

gene_anno = fread("/home/MR_rawdata/Bioherme_DEP_GWAS/gene_annotation_biomart_A.txt")
chrom = fread("/home/MR_rawdata/Bioherme_DEP_GWAS/chr_length")


for (i in 1:length(protein_List)){
 
  message(paste0("\n\tProcessing protein list ", protein_List[i], "(num :", i,") ...\n"))
  start <- Sys.time()
  
 
  pwas_file <- fread(protein_List[i])
  pwas_file=inner_join(pwas_file, snp_meta, by=c("SNP"="SNPID"))
  
  protein_ID=sub(".*gwas_(seq\\.[0-9]+\\.[0-9]+)(?:\\.[^.]+)?\\.assoc\\.linear$", "\\1", protein_List[i])
  
  pwas_file$phenotype = annot %>% filter(Protein == protein_ID) %>% pull(EntrezGeneSymbol)
  
  g <- gene_anno %>% dplyr::filter(hgnc_symbol == pwas_file$phenotype[1])
  
  if (nrow(g) == 0) {
    message("\n No gene coordinates for ", protein_ID, ". Skipping protein...\n")
    unprocessed_proteins <- rbind(unprocessed_proteins,
                                  data.frame(pro = protein_ID, reason = "No gene coordinates (GRCh38)"))
    data.table::fwrite(unprocessed_proteins, unprocessed_proteins_file, sep = "\t", append = FALSE)
    next
  }
  cis_chr   <- as.integer(g$chromosome_name)
  cis_start <- max(0L, as.integer(g$start_position) - cis_window)
  cis_end   <- (as.integer(g$end_position) + cis_window)
  
  pwas_cis <- pwas_file %>%
    dplyr::filter(CHR == cis_chr, BP >= cis_start, BP <= cis_end)
  
  if (nrow(pwas_cis) == 0) {
    message("\n No cis SNPs for ", protein_ID, ". Skipping protein...\n")
    unprocessed_proteins <- rbind(unprocessed_proteins,
                                  data.frame(pro = protein_ID, reason = "No cis SNPs after MHC exclusion"))
    data.table::fwrite(unprocessed_proteins, unprocessed_proteins_file, sep = "\t", append = FALSE)
    next
  }
  
  exp_P_format <- format_data(
    as.data.frame(pwas_cis),
    type = "exposure",
    chr_col = "CHR",
    pos_col = "BP",
    snp_col = "rsID",
    beta_col = "BETA",
    se_col = "SE",
    eaf_col = "MAF",
    effect_allele_col = "EA",
    other_allele_col = "NEA",
    pval_col = "P",
    samplesize_col = "NMISS",
    phenotype_col = "phenotype"
  )
  
  
  exp_P_clump <- ld_clump(dat = tibble(rsid=exp_P_format$SNP, pval=exp_P_format$pval.exposure, id=exp_P_format$exposure), 
                          clump_kb = 1000,
                          clump_r2 = 0.01,
                          clump_p = 1,
                          plink_bin = plinkbinr::get_plink_exe(),
                          bfile = "/home/MR_rawdata/g1000_eur/g1000_eur") 
  
  exp_P_dat <- exp_P_format[which(exp_P_format$SNP %in% exp_P_clump$rsid),]
  
  exp_P_dat <- exp_P_dat %>%
    mutate(palin = is_palindromic_vec(effect_allele.exposure,
                                      other_allele.exposure)) %>%
    filter(!(palin == TRUE & between(eaf.exposure, 0.45, 0.55))) %>%
    select(-palin)
  
  exposure_P_dat <- subset(x = exp_P_dat, subset = eaf.exposure >= eaf_threshold)
  exposure_P_dat <- subset(x = exposure_P_dat, subset = pval.exposure <= pval_threshold)
  
  
  if(nrow(exposure_P_dat) == 0){
    message(paste0("\n No SNPs remaining after pval filtering for ", protein_ID, ". Skipping protein...\n"))
    rm(list = c("pwas_file", "exp_P_format", "exp_P_clump", "exposure_P_dat"))
    unprocessed_proteins <- rbind(unprocessed_proteins, data.frame(pro = protein_ID, reason = as.character("Exposure : No SNPs remaining after pval filtering")))
    fwrite(x = unprocessed_proteins,
           file = unprocessed_proteins_file,
           sep = "\t", append = FALSE)
    next
  }
  
  ## Harmonizing ##
  dat_P <- TwoSampleMR::harmonise_data(exposure_dat = exposure_P_dat,
                                       outcome_dat = outcome_format)
  
  if (nrow(dat_P) == 0) {
    message("\n\tNo SNPs remaining after harmonization. Skipping protein...\n")
    unprocessed_proteins <- rbind(unprocessed_proteins,
                                  data.frame(pro = protein_ID, reason = "No SNPs after harmonization"))
    data.table::fwrite(unprocessed_proteins, unprocessed_proteins_file, sep = "\t", append = FALSE)
    next
  }
  
  dat_P$R2 <- dat_P$beta.exposure^2 / (dat_P$samplesize.exposure * dat_P$se.exposure^2)
  dat_P$Fval <- (dat_P$beta.exposure / dat_P$se.exposure)^2
  
  
  if(nrow(dat_P) >= 3){
    mr_pro <- mr(dat = dat_P, method_list = c("mr_ivw", "mr_weighted_median", "mr_weighted_mode", "mr_egger_regression"))
    if(nrow(mr_pro) == 0){
      unprocessed_proteins <- rbind(unprocessed_proteins, data.frame(pro = protein_ID, reason = as.character("MR : No SNPs remaining for MR")))
      message("No SNPs available for MR for ", protein_ID, "\n")
      fwrite(x = unprocessed_proteins,
             file = unprocessed_proteins_file,
             sep = "\t", append = FALSE)
      next
    }
    
    CIs <- generate_odds_ratios(mr_pro)
    
    if(sum(grepl(pattern = "Inverse variance weighted", x = CIs$method)) >= 1){
      mr_pro$b_CIlow <- subset(x = CIs, subset = method == "Inverse variance weighted", select = "lo_ci")[[1]]
      mr_pro$b_CIhigh <- subset(x = CIs, subset = method == "Inverse variance weighted", select = "up_ci")[[1]]
    } else {
      mr_pro$b_CIlow <- NA
      mr_pro$b_CIhigh <- NA
    }
    
    if(sum(grepl(pattern = "Weighted median", x = CIs$method)) >= 1){
      mr_pro$b_weightedmed_CIlow <- subset(x = CIs, subset = method == "Weighted median", select = "lo_ci")[[1]]
      mr_pro$b_weightedmed_CIhigh <- subset(x = CIs, subset = method == "Weighted median", select = "up_ci")[[1]]
    } else {
      mr_pro$b_weightedmed_CIlow <- NA
      mr_pro$b_weightedmed_CIhigh <- NA
    }
    
    if(sum(grepl(pattern = "Weighted mode", x = CIs$method)) >= 1){
      mr_pro$b_weightedmod_CIlow <- subset(x = CIs, subset = method == "Weighted mode", select = "lo_ci")[[1]]
      mr_pro$b_weightedmod_CIhigh <- subset(x = CIs, subset = method == "Weighted mode", select = "up_ci")[[1]]
    } else {
      mr_pro$b_weightedmod_CIlow <- NA
      mr_pro$b_weightedmod_CIhigh <- NA
    }
    
    if(sum(grepl(pattern = "MR Egger", x = CIs$method)) >= 1){
      mr_pro$b_egger_CIlow <- subset(x = CIs, subset = method == "MR Egger", select = "lo_ci")[[1]]
      mr_pro$b_egger_CIhigh <- subset(x = CIs, subset = method == "MR Egger", select = "up_ci")[[1]]
    } else {
      mr_pro$b_egger_CIlow <- NA
      mr_pro$b_egger_CIhigh <- NA
    }
    
    mr_egg <- mr_egger_regression(b_exp = dat_P$beta.exposure, b_out = dat_P$beta.outcome, se_exp = dat_P$se.exposure, se_out = dat_P$se.outcome)
    
    if(length(mr_egg) == 0){
      mr_pro$nsnp_egger_i <- NA
      mr_pro$b_egger_i <- NA
      mr_pro$se_egger_i <- NA
      mr_pro$pval_egger_i <- NA
    } else {
      egger_inter <- data.frame(nsnp = mr_egg$nsnp, b = mr_egg$b_i, se = mr_egg$se_i, pval = mr_egg$pval_i)
      mr_pro$nsnp_egger_i <- egger_inter$nsnp
      mr_pro$b_egger_i <- egger_inter$b
      mr_pro$se_egger_i <- egger_inter$se
      mr_pro$pval_egger_i <- egger_inter$pval
    }
    
    mr_het <- mr_heterogeneity(dat_P)
    
    if(nrow(mr_het) == 0){
      mr_pro$Q_IVW <- NA
      mr_pro$P_Q_IVW <- NA
      mr_pro$Q_Egger <- NA
      mr_pro$P_Q_Egger <- NA
    } else {
      if(sum(grepl(pattern = "Inverse variance weighted", x = mr_het$method)) >= 1){
        mr_pro$Q_IVW <- subset(x = mr_het, subset = method == "Inverse variance weighted", select = "Q")[[1]]
        mr_pro$P_Q_IVW <- subset(x = mr_het, subset = method == "Inverse variance weighted", select = "Q_pval")[[1]]
      } else {
        mr_pro$Q_IVW <- NA
        mr_pro$P_Q_IVW <- NA
      }
      if(sum(grepl(pattern = "MR Egger", x = mr_het$method)) >= 1){
        mr_pro$Q_Egger <- subset(x = mr_het, subset = method == "MR Egger", select = "Q")[[1]]
        mr_pro$P_Q_Egger <- subset(x = mr_het, subset = method == "MR Egger", select = "Q_pval")[[1]]
      } else {
        mr_pro$Q_Egger <- NA
        mr_pro$P_Q_Egger <- NA
      }
    }
    
    mr_pro.clean <- mr_pro[1,]
    
    if(sum(grepl(pattern = "MR Egger", x = mr_pro$method)) >= 1){
      mr_pro.clean$b_egger <- subset(x = mr_pro, subset = method == "MR Egger", select = "b")[[1]]
      mr_pro.clean$se_egger <- subset(x = mr_pro, subset = method == "MR Egger", select = "se")[[1]]
      mr_pro.clean$pval_egger <- subset(x = mr_pro, subset = method == "MR Egger", select = "pval")[[1]]
    } else {
      mr_pro.clean$b_egger <- NA
      mr_pro.clean$se_egger <- NA
      mr_pro.clean$pval_egger <-NA
    }
    if(sum(grepl(pattern = "Weighted median", x = mr_pro$method)) >= 1){
      mr_pro.clean$b_weighted_median <- subset(x = mr_pro, subset = method == "Weighted median", select = "b")[[1]]
      mr_pro.clean$se_weighted_median <- subset(x = mr_pro, subset = method == "Weighted median", select = "se")[[1]]
      mr_pro.clean$pval_weighted_median <- subset(x = mr_pro, subset = method == "Weighted median", select = "pval")[[1]]
    } else {
      mr_pro.clean$b_weighted_median <- NA
      mr_pro.clean$se_weighted_median <- NA
      mr_pro.clean$pval_weighted_median <- NA
    }
    if(sum(grepl(pattern = "Weighted mode", x = mr_pro$method)) >= 1){
      mr_pro.clean$b_weighted_mode <- subset(x = mr_pro, subset = method == "Weighted mode", select = "b")[[1]]
      mr_pro.clean$se_weighted_mode <- subset(x = mr_pro, subset = method == "Weighted mode", select = "se")[[1]]
      mr_pro.clean$pval_weighted_mode <- subset(x = mr_pro, subset = method == "Weighted mode", select = "pval")[[1]]
    } else {
      mr_pro.clean$b_weighted_mode <- NA
      mr_pro.clean$se_weighted_mode <- NA
      mr_pro.clean$pval_weighted_mode <- NA
    }
    mr_pro <- mr_pro.clean
  }
  
  
  if(nrow(dat_P) == 2){
    mr_pro <- mr(dat = dat_P) ##??
    if(nrow(mr_pro) == 0){
      message("No SNPs available for MR for ", protein_ID, "\n")
      unprocessed_proteins <- rbind(unprocessed_proteins, data.frame(pro = protein_ID, reason = as.character("MR : No SNPs remaining for MR")))
      fwrite(x = unprocessed_proteins,
             file = unprocessed_proteins_file,
             sep = "\t", append = FALSE)
      next
    }
    
    CIs <- generate_odds_ratios(mr_pro)
    
    mr_pro$b_CIlow <- subset(x = CIs, subset = method %in% c("Inverse variance weighted", "Wald ratio"), select = "lo_ci")[[1]]
    mr_pro$b_CIhigh <- subset(x = CIs, subset = method %in% c("Inverse variance weighted", "Wald ratio"), select = "up_ci")[[1]]
    
    mr_het <- mr_heterogeneity(dat_P)
    
    if(nrow(mr_het) == 0){
      mr_pro$Q_Egger <- NA
      mr_pro$P_Q_Egger <- NA
      mr_pro$Q_IVW <- NA
      mr_pro$P_Q_IVW <- NA
    } else {
      mr_pro$Q_Egger <- NA
      mr_pro$P_Q_Egger <- NA
      mr_pro$Q_IVW <- mr_het$Q[[1]]
      mr_pro$P_Q_IVW <- mr_het$Q_pval[[1]]
    }
  }
  
  if(nrow(dat_P) == 1){
    mr_wald <- mr_wald_ratio(b_exp = dat_P$beta.exposure, b_out = dat_P$beta.outcome, se_exp = dat_P$se.exposure, se_out = dat_P$se.outcome)
    if(length(mr_wald) == 0){
      message("No SNPs available for MR for ", protein_ID, "\n")
      unprocessed_proteins <- rbind(unprocessed_proteins, data.frame(pro = protein_ID, reason = as.character("MR : No SNPs remaining for MR")))
      fwrite(x = unprocessed_proteins,
             file = unprocessed_proteins_file,
             sep = "\t", append = FALSE)
      next
    }
    CIs <- generate_odds_ratios(mr_wald)
    mr_pro <- data.frame(outcome=outcome_format$outcome[1], exposure = exposure_P_dat$exposure[1], method="Wald ratio", nsnp = mr_wald$nsnp, b = mr_wald$b, b_CIlow = CIs$lo_ci, b_CIhigh = CIs$up_ci, se = mr_wald$se, pval = mr_wald$pval)
    
    mr_pro$id.exposure <- NA
    mr_pro$id.outcome <- NA
    mr_pro$Q_Egger <- NA
    mr_pro$P_Q_Egger <- NA
    mr_pro$Q_IVW <- NA
    mr_pro$P_Q_IVW <- NA
  }
  
  
  if((nrow(dat_P) == 2) | (nrow(dat_P) == 1)){
    mr_pro$b_egger <- NA
    mr_pro$se_egger <- NA
    mr_pro$pval_egger <- NA
    
    mr_pro$b_weighted_median <- NA
    mr_pro$se_weighted_median <- NA
    mr_pro$pval_weighted_median <- NA
    
    mr_pro$b_weighted_mode <- NA
    mr_pro$se_weighted_mode <- NA
    mr_pro$pval_weighted_mode <- NA
    
    mr_pro$b_egger_CIlow <- NA
    mr_pro$b_egger_CIhigh <- NA
    mr_pro$b_weightedmed_CIlow <- NA
    mr_pro$b_weightedmed_CIhigh <- NA
    mr_pro$b_weightedmod_CIlow <- NA
    mr_pro$b_weightedmod_CIhigh <- NA
    
    mr_pro$nsnp_egger_i <- NA
    mr_pro$b_egger_i <- NA
    mr_pro$se_egger_i <- NA
    mr_pro$pval_egger_i <- NA
  }
  
  ## Saving results ####
  mr_res <- rbind(mr_res, mr_pro)
  fwrite(x = mr_res,
         file = "/home/MR_rawdata/Bioherme_DEP_GWAS/MR_results/cis_proteins_outcome.txt",
         sep = "\t", append = FALSE)
}  

