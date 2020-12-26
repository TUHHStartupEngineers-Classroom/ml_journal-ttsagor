# LIME FEATURE EXPLANATION ----

# 1. Setup ----

# Load Libraries 
library(h2o)
library(recipes)
library(readxl)
library(tidyverse)
library(tidyquant)
library(lime)
library(tidymodels)
library(magrittr)
library(dplyr)
library(sjmisc)
library(haven)
library(sjlabelled)
library(rsample)
library(rstanarm)
library(broom.mixed)
library(stringr)
library(data.table)
library(cowplot)
library(glue)
library(ggplot2)

# Load Data
employee_attrition_tbl <- read_csv("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/datasets-1067-1925-WA_Fn-UseC_-HR-Employee-Attrition.csv")
definitions_raw_tbl    <- read_excel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/data_definitions.xlsx", sheet = 1, col_names = FALSE)

# Processing Pipeline
process_hr_data_readable <- function(data, definitions_tbl) {
  
  definitions_list <- definitions_tbl %>%
    fill(...1, .direction = "down") %>%
    filter(!is.na(...2)) %>%
    separate(...2, into = c("key", "value"), sep = " '", remove = TRUE) %>%
    rename(column_name = ...1) %>%
    mutate(key = as.numeric(key)) %>%
    mutate(value = value %>% str_replace(pattern = "'", replacement = "")) %>%
    split(.$column_name) %>%
    map(~ select(., -column_name)) %>%
    map(~ mutate(., value = as_factor(value))) 
  
  for (i in seq_along(definitions_list)) {
    list_name <- names(definitions_list)[i]
    colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
  }
  
  data_merged_tbl <- list(HR_Data = data) %>%
    append(definitions_list, after = 1) %>%
    reduce(left_join) %>%
    select(-one_of(names(definitions_list))) %>%
    set_names(str_replace_all(names(.), pattern = "_value", 
                              replacement = "")) %>%
    select(sort(names(.))) %>%
    mutate_if(is.character, as.factor) %>%
    mutate(
      BusinessTravel = BusinessTravel %>% fct_relevel("Non-Travel", 
                                                      "Travel_Rarely", 
                                                      "Travel_Frequently"),
      MaritalStatus  = MaritalStatus %>% fct_relevel("Single", 
                                                     "Married", 
                                                     "Divorced")
    )
  
  return(data_merged_tbl)
  
}
process_hr_data_readable(employee_attrition_tbl, definitions_raw_tbl) %>% 
  glimpse()

employee_attrition_readable_tbl <- process_hr_data_readable(employee_attrition_tbl, definitions_raw_tbl)

# Split into test and train
set.seed(seed = 1113)
split_obj <- rsample::initial_split(employee_attrition_readable_tbl, prop = 0.75)

# Assign training and test data
train_readable_tbl <- training(split_obj)
test_readable_tbl  <- testing(split_obj)

# ML Preprocessing Recipe 
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
  step_zv(all_predictors()) %>%
  step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>% 
  prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

# 2. Models ----

h2o.init()

automl_leader <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/StackedEnsemble_BestOfFamily_AutoML_20201226_190932")
automl_leader



# 3. LIME ----

# 3.1 Making Predictions ----

predictions_tbl <- automl_leader %>% 
  h2o.predict(newdata = as.h2o(test_tbl)) %>%
  as.tibble() %>%
  bind_cols(
    test_tbl %>%
      select(Attrition, EmployeeNumber)
  )

predictions_tbl






explainer <- train_tbl %>%
  select(-Attrition) %>%
  lime(
    model           = automl_leader,
    bin_continuous  = TRUE,
    n_bins          = 4,
    quantile_bins   = TRUE
  )

explainer




explanation <- test_tbl %>%
  slice(1) %>%
  select(-Attrition) %>%
  lime::explain(
    
    # Pass our explainer object
    explainer = explainer,
    # Because it is a binary classification model: 1
    n_labels   = 1,
    # number of features to be returned
    n_features = 8,
    # number of localized linear models
    n_permutations = 5000,
    # Let's start with 1
    kernel_width   = 1
  )

explanation




explanation %>%
  as.tibble() %>%
  select(feature:prediction) 


g <- plot_features(explanation = explanation, ncol = 1)


# 3.3 Multiple Explanations ----

explanation <- test_tbl %>%
  slice(1:20) %>%
  select(-Attrition) %>%
  lime::explain(
    explainer = explainer,
    n_labels   = 1,
    n_features = 8,
    n_permutations = 5000,
    kernel_width   = 0.5
  )

explanation %>%
  as.tibble()

plot_features(explanation, ncol = 4)

plot_explanations(explanation)

case_1 <- explanation %>%
  filter(case == 1)

case_1 %>%
  plot_features() 



  
# With ggplot
p<-ggplot(data=case_1, aes(x=feature_desc, y=feature_weight, fill = ifelse(feature_weight<0, 'blue', 'red'))) +
  geom_bar(stat="identity")  + 
  labs(
    title = "case:1",
    subtitle = "label:yes",
    x="Features",
    y="Weight",
    fill =""
  ) +
  scale_fill_hue(labels = c("contradicts", "supports")) + coord_flip()
p

# heat map
level_order <- factor(explanation$case, level = c('1', '2', '3','4','5','6','7', '8','9','10','11','12', '13','14','15','16','17', '18','19','20'))

h <- ggplot(explanation, aes(x=level_order, y=feature_desc, fill= feature_weight)) + 
  scale_x_discrete(drop = FALSE) + 
  labs(
    x="Case",
    y="Feature",
    fill ="Feature weight"
  ) + 
  geom_tile()  + 
  facet_wrap(~ label)

h


