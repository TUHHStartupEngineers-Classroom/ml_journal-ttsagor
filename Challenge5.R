library(tidymodels)
library(magrittr)
library(dplyr)
library(sjmisc)
library(magrittr)
library(haven)
library(sjlabelled)
library(rsample)
library(recipes)
library(rstanarm)
library(broom.mixed)
library(h2o)

product_backorders_tbl <- read.csv("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/product_backorders.csv")

product_backorders_tbl %>% glimpse()

data_split <- initial_split(product_backorders_tbl, prop = 3/4)
# Assign training and test data
train_data <- training(data_split)
test_data  <- testing(data_split)

factor_names <- c("went_on_backorder")
product_rec <- 
  recipe(went_on_backorder ~ ., data = train_data) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_mutate_at(factor_names, fn = as.factor) %>%
  prep()
d <- summary(product_rec)
train_tbl <- bake(product_rec, new_data = train_data)
test_tbl  <- bake(product_rec, new_data = test_data)


# Modeling
h2o.init()

# Split data into a training and a validation data frame
# Setting the seed is just for reproducability
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.75), seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

# Set the target and predictors
y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)


automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 30,
  nfolds            = 5 
)


typeof(automl_models_h2o)
slotNames(automl_models_h2o)
automl_models_h2o@leaderboard 
automl_models_h2o@leader

h2o.getModel("DRF_1_AutoML_20201224_182023")

extract_h2o_model_name_by_position <- function(h2o_leaderboard, n = 1, verbose = T) {
  
  model_name <- h2o_leaderboard %>%
    as_tibble() %>%
    slice(n) %>%
    pull(model_id)
  
  if (verbose) message(model_name)
  
  return(model_name)
  
}

automl_models_h2o@leaderboard %>% 
  extract_h2o_model_name_by_position(6) %>% 
  h2o.getModel() %>% 
  h2o.saveModel(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/")



stacked_ensemble_h2o <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_2_AutoML_20201224_183739")
stacked_ensemble_h2o

predictions <- h2o.predict(stacked_ensemble_h2o, newdata = as.h2o(test_tbl))

typeof(predictions)

predictions_tbl <- predictions %>% as_tibble()
predictions_tbl
