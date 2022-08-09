#### start-up and script variables ####
data_name = "lungcancer"
raw_data_file = "lungcancer_raw.csv";
outcome_name = "Malignant"
partition_names = c('Gender', 'Age')
sampling_weight_name = "sampling_weights"
ordinal_names = c("");

#formats
raw_missing_label = "?";

#set directories
data_dir = "/Users/username/XXX/repos/fair_decoupling/data/"
raw_data_dir = paste0(data_dir, data_name, "/")

#output file names
format_label = "envyfree"
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
to_drop = apply(raw_df == "Missing",MARGIN = 1, any)
raw_df = raw_df[!to_drop,]
raw_df = raw_df[complete.cases(raw_df), ]

df = raw_df %>%
    filter(gender %in% c('Male', 'Female')) %>%
    mutate(Age = ifelse(age<60, '<60', '60+'),
           Gender = ifelse(gender =='Male', 'Male', 'Female')) %>%
    select(Gender,
           Age,
           FamilyHistory=family.history,
           LobeLocation=lobe.location,
           NumNodules=X..of.nodules,
           NoduleType=nodule.type,
           NoduleSize=nodule.size,
           Margin=margin,
           Emphysema=emphsema,
           Malignant=malignant) %>%
    filter(FamilyHistory != "Missing",
           !is.na(Emphysema),
           Malignant != 'ce') %>%
    mutate(FamilyHistory = ifelse(FamilyHistory ==0,0,1),
           Malignant = ifelse(Malignant==0, FALSE, TRUE),
           Age = as.factor(Age),
           Gender = as.factor(Gender))



df = df %>%
    mutate(SpeculatedMargin = Margin==2,
           #
           NumNodules_eq_1=NumNodules==1,
           NumNodules_geq_2=NumNodules>=2,
           #
           NoduleType_is_1=NoduleType==1,
           NoduleType_is_2=NoduleType==2,
           NoduleType_is_3=NoduleType==3,
           #
           NoduleSize_lt_7mm=NoduleSize<7,
           NoduleSize_geq_7mm=NoduleSize>=7,
           NoduleSize_geq_12mm=NoduleSize>=12,
           NoduleSize_geq_18mm=NoduleSize>=18,
           #
           LobeLocationIsRightUpper = LobeLocation==1,
           LobeLocationIsRightMiddle = LobeLocation==2,
           LobeLocationIsRightLower = LobeLocation==3,
           LobeLocationIsLeftUpper = LobeLocation==4,
           LobeLocationIsLingula = LobeLocation==5,
           LobeLocationIsLeftLower = LobeLocation==6) %>%
    select(-LobeLocation,
           -NumNodules,
           -NoduleType,
           -NoduleSize,
           -Margin)
df$FamilyHistory = as.logical(df$FamilyHistory)
df$Emphysema = as.logical(df$Emphysema)

data = df

#save

data = data %>% select(outcome_name, partition_names, everything())

#raw data
write.csv(x = data, file = data_file, row.names = FALSE, quote = FALSE);

#helper file
helper_df = get.header.descriptions(data, outcome_name = outcome_name, partition_names = partition_names);
write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE)


