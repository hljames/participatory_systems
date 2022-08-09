#### start-up and script variables ####
data_name = "apnea"
raw_data_file = "apnea_raw.csv";
outcome_name = "Apnea"
partition_names = c('Age', 'Sex')
sampling_weight_name = "sampling_weights"
ordinal_names = c("");


#set directories
data_dir = "/Users/username/XXX/repos/psc/data/"
raw_data_dir = paste0(data_dir, data_name, "/")

#output file names
file_header = paste0(raw_data_dir, data_name)
data_file = paste0(file_header, "_data.csv")
helper_file = paste0(file_header, "_helper.csv")
weights_file = paste0(file_header, "_weights.csv");

#load libraries
source(paste0(data_dir,"processing_functions.R"));
required_packages = c('dplyr', 'tidyr', 'forcats','stringr')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

raw_df = read.csv(paste0(raw_data_dir, raw_data_file))
df = raw_df %>%
    mutate(Sex = as.factor(ifelse(female, 'Female', 'Male')),
           Age = as.factor(ifelse(age_lt_30, '<30',  no = ifelse(age_geq_60, '60+', '30_to_60')))) %>%
    select(-female, -starts_with('age_'), -uars, -osa, -plmi_ge_10, -cai_ge_5,-ESS_sum_geq_11) %>%
    select(Apnea = osa_or_uars, partition_names, everything())

data = df

#save

data = data %>% select(outcome_name, partition_names, everything())

#raw data
write.csv(x = data, file = data_file, row.names = FALSE, quote = FALSE);

#helper file
helper_df = get.header.descriptions(data, outcome_name = outcome_name, partition_names = partition_names)
write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE)

