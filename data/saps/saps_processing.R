#### start-up and script variables ####
data_name = "saps"
raw_data_file = "SAPS_Raw_Data.csv";
outcome_name = "DeadAtDischarge"
partition_names = c('Age', 'HIVWithComplications')
sampling_weight_name = "sampling_weights"
ordinal_names = c("");


#set directories
data_dir = "/Users/vinithms/Documents/fairusehc/data/"
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

#description of features
raw_df = read.csv(paste0(raw_data_dir, raw_data_file))
raw_df <- raw_df[,colSums(is.na(raw_df))<nrow(raw_df)]

raw_df[is.na(raw_df)] = 0

df = raw_df %>%
    mutate(Age = as.factor(ifelse(Age < 30, '<30',  '>30')),
           #
           HighestTemp_leq_34 = between(HighestBodyTemperature, 0, 34),
           HighestTemp_leq_35 = between(HighestBodyTemperature, 0, 35),
           HighestTemp_geq_39 = between(HighestBodyTemperature, 39, 50),
           HighestTemp_geq_40 = between(HighestBodyTemperature, 40, 50),
           #
           GlasgowComaScore_geq_9 = GlasgowComaScore >= 9,
           GlasgowComaScore_geq_13 = GlasgowComaScore >= 13,
           #
           WhiteBloodCellCount_leq_3BpL = WhiteBloodCellCount <= 3,
           WhiteBloodCellCount_geq_11BpL = WhiteBloodCellCount >= 11,
           #
           SystolicBP_geq_140mmHg = SystolicBloodPressure >= 140,
           SystolicBP_geq_160mmHg = SystolicBloodPressure >= 160,
           SystolicBP_geq_180mmHg = SystolicBloodPressure >= 180,
           #
           DailyUrinaryOutput_leq_500ml = UrinaryOutput <= 0.5,
           DailyUrinaryOutput_geq_3000ml = UrinaryOutput >= 3.0,
           #
           #https://www.nlm.nih.gov/medlineplus/ency/article/003481.htm
           #The normal range for blood sodium levels is 135 to 145 mEq/L
           SerumSodiumLevel_leq_135mEqL = SerumSodiumLevel <= 135,
           SerumSodiumLevel_geq_145mEqL = SerumSodiumLevel >= 145,
           #
           #https://www.nlm.nih.gov/medlineplus/ency/article/003484.htm
           #The normal range is 3.7 to 5.2 mEq/L.
           #
           SerumPotassiumLevel_leq_3o7mEqL = SerumPotassiumLevel <= 3.7,
           SerumPotassiumLevel_geq_5o2mEqL = SerumPotassiumLevel >= 5.2,
           #
           # http://www.mayomedicallaboratories.com/test-catalog/Clinical+and+Interpretive/876
           # Males and Females > or =18 years: 22-29 mmol/L
           # Note: ranges are smaller for younger adults (we ignore this)
           #
           SerumBicarbonateLevel_leq_21mmolL = SerumBicarbonateLevel <= 21,
           SerumBicarbonateLevel_geq_30mmolL = SerumBicarbonateLevel >= 30,
           #
           #http://www.mayoclinic.org/tests-procedures/blood-urea-nitrogen/basics/results/prc-20020239
           #In general, 7 to 20 mg/dL (2.5 to 7.1 mmol/L) is considered normal.
           SerumUreaNitrogenLevelLevel_leq_2o5_mmolL = SerumUreaNitrogenLevel <= 2.5,
           SerumUreaNitrogenLevelLevel_geq_7o1_mmolL = SerumUreaNitrogenLevel >= 7.1,
           #
           AnyOrganSystemFailure = TotalOrganSystemFailure > 0,
           MultipleOrganSystemFailure = TotalOrganSystemFailure > 1) %>%
    select(-Country,
           -Service,
           -Number,
           -AdmissionType,
           -HeartRate,
           -starts_with("VentilationArteryPressureRatio"),
           -UrinaryOutput,
           -SystolicBloodPressure,
           -HighestBodyTemperature,
           -WhiteBloodCellCount,
           -SerumPotassiumLevel,
           -SerumSodiumLevel,
           -SerumBicarbonateLevel,
           -BillirubinLevel,
           -SerumUreaLevel,
           -SerumUreaNitrogenLevel,
           -GlasgowComaScore,
           -SPROBMT)

#replace NA with 0


df = df %>%
    select(DeadAtDischarge,
           partition_names,
           everything()) %>%
    mutate(DeadAtDischarge = as.numeric(DeadAtDischarge),
           HIVWithComplications = as.factor(ifelse(HIVWithComplications, "Positive", "Negative")),
           GlasgowComaScore_geq_9 = as.numeric(GlasgowComaScore_geq_9),
           GlasgowComaScore_geq_13 = as.numeric(GlasgowComaScore_geq_13),
           AnyOrganSystemFailure = as.numeric(AnyOrganSystemFailure),
           MultipleOrganSystemFailure = as.numeric(MultipleOrganSystemFailure))

df = convert.logical.to.binary(df)


write.csv(x = df, file =data_file, row.names = FALSE, quote = FALSE)
#helper file
helper_df = get.header.descriptions(df, outcome_name = outcome_name, partition_names = partition_names);
write.csv(x = helper_df, file = helper_file, row.names = FALSE, quote = FALSE)





