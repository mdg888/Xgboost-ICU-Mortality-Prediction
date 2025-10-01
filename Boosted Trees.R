# Load required libraries (keeping your existing library loads)
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(yardstick))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(tidymodels))
suppressPackageStartupMessages(library(discrim))
library(fastDummies)
library(xtable)
library(knitr)
library(ggplot2)
library(tidyr)
library(lubridate)
library(vip)

# [Keep all your existing data loading and preprocessing code from before]
# Load data
training_data_mimic_x <- read_csv("C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/mimic_data_kaggle/mimic_train_X.csv", show_col_types = F)
training_data_mimic_y <- read_csv("C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/mimic_data_kaggle/mimic_train_y.csv", show_col_types = F)
test_data_mimic_y <- read_csv("C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/mimic_data_kaggle/mimic_kaggle_death_sample_submission.csv", show_col_types = F)
test_data_mimic_x <- read_csv("C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/mimic_data_kaggle/mimic_test_X.csv", show_col_types = F)
extra_data <- read_csv("C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/mimic_data_kaggle/extra_data/MIMIC_diagnoses.csv", show_col_types = F)

# Remove ICD9_diagnosis column from the X datasets if it exists (we'll use extra_data instead)
if("ICD9_diagnosis" %in% names(training_data_mimic_x)) {
  training_data_mimic_x <- training_data_mimic_x %>% select(-ICD9_diagnosis)
}
if("ICD9_diagnosis" %in% names(test_data_mimic_x)) {
  test_data_mimic_x <- test_data_mimic_x %>% select(-ICD9_diagnosis)
}

# Rename the target variable in y datasets for clarity
training_data_mimic_y <- training_data_mimic_y %>%
  rename(Death = HOSPITAL_EXPIRE_FLAG) %>%
  select(-any_of("...1"))

test_data_mimic_y <- test_data_mimic_y %>%
  rename(Death = HOSPITAL_EXPIRE_FLAG, icustay_id = ID)

# Function to process ICD-9 diagnosis data
# Enhanced function to process ICD-9 diagnosis data with proper hospital stay counting
# CORRECTED process_icd9_data function that properly handles hospital stays
process_icd9_data <- function(extra_data, main_data) {
  cat("Processing ICD-9 diagnosis data...\n")
  
  # First, create patient-level hospital stay counts
  hospital_stays <- extra_data %>%
    group_by(SUBJECT_ID) %>%
    summarise(total_hospital_stays = n_distinct(HADM_ID), .groups = "drop")
  
  # Process ICD-9 codes and create features
  icd9_processed <- extra_data %>%
    mutate(
      # Extract first 3 characters for categorization
      icd9_first3 = substr(ICD9_CODE, 1, 3),
      
      # Create main category groupings based on ICD-9 structure
      icd9_category = case_when(
        # E codes (external causes)
        substr(icd9_first3, 1, 1) == "E" ~ "external_causes",
        # V codes (supplementary classification)
        substr(icd9_first3, 1, 1) == "V" ~ "supplementary",
        # Convert to numeric and categorize, using safe conversion
        TRUE ~ {
          numeric_code <- suppressWarnings(as.numeric(icd9_first3))
          case_when(
            is.na(numeric_code) ~ "other",
            numeric_code >= 1 & numeric_code <= 139 ~ "infectious",
            numeric_code >= 140 & numeric_code <= 239 ~ "neoplasms",
            numeric_code >= 240 & numeric_code <= 279 ~ "endocrine",
            numeric_code >= 280 & numeric_code <= 289 ~ "blood",
            numeric_code >= 290 & numeric_code <= 319 ~ "mental",
            numeric_code >= 320 & numeric_code <= 389 ~ "nervous",
            numeric_code >= 390 & numeric_code <= 459 ~ "circulatory",
            numeric_code >= 460 & numeric_code <= 519 ~ "respiratory",
            numeric_code >= 520 & numeric_code <= 579 ~ "digestive",
            numeric_code >= 580 & numeric_code <= 629 ~ "genitourinary",
            numeric_code >= 630 & numeric_code <= 679 ~ "pregnancy",
            numeric_code >= 680 & numeric_code <= 709 ~ "skin",
            numeric_code >= 710 & numeric_code <= 739 ~ "musculoskeletal",
            numeric_code >= 740 & numeric_code <= 759 ~ "congenital",
            numeric_code >= 760 & numeric_code <= 779 ~ "perinatal",
            numeric_code >= 780 & numeric_code <= 799 ~ "symptoms",
            numeric_code >= 800 & numeric_code <= 999 ~ "injury",
            TRUE ~ "other"
          )
        }
      ),
      
      # Create severity indicator based on sequence number
      high_priority = ifelse(SEQ_NUM <= 3, 1, 0)
    )
  
  # Create summary features per patient-admission combination
  icd9_summary <- icd9_processed %>%
    group_by(SUBJECT_ID, HADM_ID) %>%
    summarise(
      # Count of diagnoses
      total_diagnoses = n(),
      high_priority_diagnoses = sum(high_priority),
      total_hospital_stays = n_distinct(HADM_ID),  # CHANGED: was distinct(HADM_ID)
      
      # Category presence indicators
      has_infectious = as.numeric(any(icd9_category == "infectious")),
      has_neoplasms = as.numeric(any(icd9_category == "neoplasms")),
      has_endocrine = as.numeric(any(icd9_category == "endocrine")),
      has_blood = as.numeric(any(icd9_category == "blood")),
      has_mental = as.numeric(any(icd9_category == "mental")),
      has_nervous = as.numeric(any(icd9_category == "nervous")),
      has_circulatory = as.numeric(any(icd9_category == "circulatory")),
      has_respiratory = as.numeric(any(icd9_category == "respiratory")),
      has_digestive = as.numeric(any(icd9_category == "digestive")),
      has_genitourinary = as.numeric(any(icd9_category == "genitourinary")),
      has_pregnancy = as.numeric(any(icd9_category == "pregnancy")),
      has_skin = as.numeric(any(icd9_category == "skin")),
      has_musculoskeletal = as.numeric(any(icd9_category == "musculoskeletal")),
      has_congenital = as.numeric(any(icd9_category == "congenital")),
      has_perinatal = as.numeric(any(icd9_category == "perinatal")),
      has_symptoms = as.numeric(any(icd9_category == "symptoms")),
      has_injury = as.numeric(any(icd9_category == "injury")),
      has_external_causes = as.numeric(any(icd9_category == "external_causes")),
      has_supplementary = as.numeric(any(icd9_category == "supplementary")),
      .groups = "drop"
    ) %>%
    # Add hospital stay counts
    left_join(hospital_stays, by = "SUBJECT_ID")
  
  # Merge with main data
  main_data_with_icd9 <- main_data %>%
    left_join(icd9_summary, by = c("subject_id" = "SUBJECT_ID", "hadm_id" = "HADM_ID"))
  
  # Fill missing values with 0 for ICD-9 features (patients without diagnoses data)
  icd9_cols <- names(icd9_summary)[!names(icd9_summary) %in% c("SUBJECT_ID", "HADM_ID")]
  main_data_with_icd9[icd9_cols] <- lapply(main_data_with_icd9[icd9_cols], function(x) ifelse(is.na(x), 0, x))
  
  return(main_data_with_icd9)
}


# Integrate ICD-9 data with main datasets
training_data_mimic_x <- process_icd9_data(extra_data, training_data_mimic_x)
test_data_mimic_x <- process_icd9_data(extra_data, test_data_mimic_x)

# Simplified preprocessing function without standardization
preprocess_mimic_data <- function(data) {
  # Save the ID column for later
  id_col <- data %>% select(icustay_id)
  
  # Remove unnecessary columns but keep subject_id and hadm_id until after ICD-9 processing
  # IMPORTANT: Remove ICD9_diagnosis if it exists to avoid the factor level error
  remove_vec <- c("...1", "subject_id", "hadm_id", "DIAGNOSIS", "Diff", "ICD9_diagnosis")
  df_processed <- data %>% select(-any_of(remove_vec))
  
  # Create date-based variables and calculate age
  df_processed <- df_processed %>% 
    mutate(
      Addtime = ifelse(!is.na(ADMITTIME), year(as.Date(ADMITTIME)), NA),
      dob = ifelse(!is.na(DOB), year(as.Date(DOB)), NA),
      Age = Addtime - dob
    ) %>%
    select(-any_of(c("ADMITTIME", "DOB", "Addtime", "dob")))
  
  # Convert GENDER to binary
  df_processed <- df_processed %>% 
    mutate(GENDER = ifelse(GENDER == "M", 1, 0))
  
  # Create variability features if min/max columns exist
  vital_signs <- c("HeartRate", "SysBP", "DiasBP", "MeanBP", "RespRate", "TempC", "SpO2", "Glucose")
  
  for(vital in vital_signs) {
    min_col <- paste0(vital, "_Min")
    max_col <- paste0(vital, "_Max")
    var_col <- paste0(vital, "_Variability")
    
    if(all(c(min_col, max_col) %in% names(df_processed))) {
      df_processed[[var_col]] <- df_processed[[max_col]] - df_processed[[min_col]]
      df_processed[[var_col]][is.na(df_processed[[var_col]])] <- 0
      df_processed <- df_processed %>% select(-all_of(c(min_col, max_col)))
    }
  }
  
  # Process categorical variables and create dummies
  categorical_cols <- c("RELIGION", "ADMISSION_TYPE", "ETHNICITY", 
                        "INSURANCE", "FIRST_CAREUNIT", "MARITAL_STATUS")
  
  for (col in categorical_cols) {
    if (col %in% names(df_processed)) {
      
      # Create dummy variables
      df_processed <- dummy_cols(df_processed, 
                                 select_columns = col,
                                 remove_selected_columns = TRUE)
    }
  }
  
  return(list(data = df_processed, id_col = id_col))
}

# Process data
cat("Processing training data with ICD-9 features...\n")
train_processed <- preprocess_mimic_data(training_data_mimic_x)
train_data <- train_processed$data
train_ids <- train_processed$id_col

cat("Processing test data with ICD-9 features...\n")
test_processed <- preprocess_mimic_data(test_data_mimic_x)
test_data <- test_processed$data
test_ids <- test_processed$id_col

# Join the processed X data with the y (target) data
train_data_with_target <- train_data %>%
  left_join(training_data_mimic_y, by = "icustay_id")

test_data_with_target <- test_data %>%
  left_join(test_data_mimic_y, by = "icustay_id")

# Final data cleaning - handle any remaining missing values
train_data_with_target <- train_data_with_target %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

test_data_with_target <- test_data_with_target %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Convert Death to factor for classification
train_data_with_target <- train_data_with_target %>%
  mutate(Death = factor(Death))

test_data_with_target <- test_data_with_target %>%
  mutate(Death = factor(Death))

# Check the distribution of Death values
cat("Distribution of Death in training data:\n")
print(table(train_data_with_target$Death, useNA = "ifany"))

# 1. EXAMINE CLASS DISTRIBUTION
cat("Examining class distribution:\n")
class_dist <- train_data_with_target %>%
  count(Death) %>%
  mutate(proportion = n/sum(n))
print(class_dist)

# Calculate class weights for imbalance
total_samples <- nrow(train_data_with_target)
pos_samples <- sum(train_data_with_target$Death == "1")
neg_samples <- sum(train_data_with_target$Death == "0")

# Calculate scale_pos_weight for XGBoost (ratio of negative to positive samples)
scale_pos_weight <- neg_samples / pos_samples

cat("Class distribution - Positive (Death=1):", pos_samples, ", Negative (Death=0):", neg_samples, "\n")
cat("Scale pos weight for XGBoost:", scale_pos_weight, "\n")

# 2. SET UP BOOSTED TREE MODEL
bt_spec_balanced <- boost_tree(
  trees = 1500,
  min_n = 15,
  tree_depth = 8,
  learn_rate = 0.02,  # Lower learning rate with more trees
  loss_reduction = 0.05,
  sample_size = 0.8
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", 
             objective = "binary:logistic",
             eval_metric = "auc",
             early_stopping = 100)

# Workflow - just basic recipe
wf_balanced <- workflow() %>%
  add_model(bt_spec_balanced) %>%
  add_recipe(
    recipe(Death ~ ., data = train_data_with_target) %>%
      update_role(icustay_id, new_role = "id")
  )

# 3. CROSS-VALIDATION SETUP
set.seed(1110)
cv_folds <- vfold_cv(train_data_with_target, v = 5, strata = Death)

# Add comprehensive evaluation metrics
custom_metrics <- metric_set(
  accuracy, yardstick::roc_auc, sensitivity, specificity, 
  f_meas, precision, recall, mcc,  # Matthew's correlation coefficient
  bal_accuracy, kap  # Kappa statistic
)

cat("Testing Balanced XGBoost...\n")
cv_results_balanced <- wf_balanced %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = custom_metrics,
    control = control_resamples(save_pred = TRUE, verbose = T)
  )

# 4. EVALUATE RESULTS
metrics_balanced <- collect_metrics(cv_results_balanced)
cat("\nBalanced XGBoost Results:\n")
print(metrics_balanced %>% select(.metric, mean, std_err))

# 5. THRESHOLD OPTIMIZATION
# Get predictions from the model
cv_predictions <- collect_predictions(cv_results_balanced)

# Function to optimize threshold based on different criteria
optimize_threshold <- function(predictions, criteria = "youden") {
  # Calculate metrics for different thresholds
  thresholds <- seq(0.1, 0.9, by = 0.05)
  threshold_results <- map_dfr(thresholds, function(thresh) {
    pred_class <- factor(ifelse(predictions$.pred_1 >= thresh, "1", "0"), levels = c("0", "1"))
    
    sens <- sensitivity_vec(predictions$Death, pred_class)
    spec <- specificity_vec(predictions$Death, pred_class)
    acc <- accuracy_vec(predictions$Death, pred_class)
    f1 <- f_meas_vec(predictions$Death, pred_class)
    
    tibble(
      threshold = thresh,
      sensitivity = sens,
      specificity = spec,
      accuracy = acc,
      f1_score = f1,
      youden = sens + spec - 1,  # Youden's J statistic
      balanced_acc = (sens + spec) / 2
    )
  })
  
  if(criteria == "youden") {
    optimal <- threshold_results[which.max(threshold_results$youden), ]
  } else if(criteria == "f1") {
    optimal <- threshold_results[which.max(threshold_results$f1_score), ]
  } else if(criteria == "balanced") {
    optimal <- threshold_results[which.max(threshold_results$balanced_acc), ]
  }
  
  return(list(optimal = optimal, all_results = threshold_results))
}

# Find optimal thresholds using different criteria
opt_youden <- optimize_threshold(cv_predictions, "youden")
opt_f1 <- optimize_threshold(cv_predictions, "f1")
opt_balanced <- optimize_threshold(cv_predictions, "balanced")

# 6. TRAIN FINAL MODEL AND MAKE PREDICTIONS
final_model <- wf_balanced %>% fit(data = train_data_with_target)

# Make predictions with optimal threshold
test_pred_prob <- predict(final_model, test_data_with_target, type = "prob")

# Use Youden's optimal threshold for final predictions
optimal_threshold <- opt_youden$optimal$threshold
test_pred_class_optimal <- factor(
  ifelse(test_pred_prob$.pred_1 >= optimal_threshold, "1", "0"), 
  levels = c("0", "1")
)

# Combine results
test_results <- test_data_with_target %>%
  select(icustay_id, Death) %>%
  bind_cols(
    predicted_class = test_pred_class_optimal,
    death_probability = test_pred_prob$.pred_1,
    survival_probability = test_pred_prob$.pred_0
  )

# 7. VISUALISATIONS
# ROC Curve
roc_data <- cv_predictions %>% roc_curve(Death, .pred_1)
roc_auc <- cv_predictions %>% roc_auc(Death, .pred_1) %>% pull(.estimate)

roc_plot <- roc_data %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(size = 1.2, color = "steelblue") +
  geom_abline(linetype = "dashed", color = "gray") +
  geom_point(data = opt_youden$optimal, 
             aes(x = 1 - specificity, y = sensitivity), 
             color = "red", size = 3) +
  annotate("text", x = 0.7, y = 0.3, 
           label = paste("Optimal Threshold:", round(optimal_threshold, 2)), 
           color = "red") +
  labs(
    title = "ROC Curve - Balanced XGBoost",
    subtitle = paste0("AUC: ", round(roc_auc, 3)),
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal() +
  coord_equal()

print(roc_plot)

# Feature importance
vip_plot <- vip(final_model, num_features = 20) +
  theme_minimal() +
  labs(title = "Feature Importance - Balanced XGBoost",
       subtitle = paste0("CV AUC: ", round(roc_auc, 3), 
                         " | Optimal Threshold: ", round(optimal_threshold, 2))) +
  theme(axis.text.y = element_text(size = 8))

print(vip_plot)

# 8. CREATE SUBMISSION FILE
submission_results <- test_results %>%
  select(icustay_id, death_probability) %>%
  rename(ID = icustay_id, HOSPITAL_EXPIRE_FLAG = death_probability)

write_csv(submission_results, "C:/Users/micha/Desktop/Uni/S1 2025/ETC3250/Project/boosted_tree_submission.csv")