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
library(stringr)
library(data.table)
library(cowplot)
library(glue)

product_backorders_tbl <- read.csv("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/product_backorders.csv")

product_backorders_tbl %>% glimpse()

data_split <- initial_split(product_backorders_tbl, prop = 3/4)
train_data <- training(data_split)
test_data  <- testing(data_split)

product_rec <- 
  recipe(went_on_backorder ~ ., data = train_data) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  prep()

train_tbl <- bake(product_rec, new_data = train_data)
test_tbl  <- bake(product_rec, new_data = test_data)


h2o.init()

split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

y <- "stop_auto_buy_Yes"
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

h2o.getModel("DeepLearning_grid__1_AutoML_20200820_190823_model_1")

extract_h2o_model_name_by_position <- function(h2o_leaderboard, n = 1, verbose = T) {
  
  model_name <- h2o_leaderboard %>%
    as.tibble() %>%
    slice(n) %>%
    pull(model_id)
  
  if (verbose) message(model_name)
  
  return(model_name)
  
}

automl_models_h2o@leaderboard %>% 
  extract_h2o_model_name_by_position(6) %>% 
  h2o.getModel() %>% 
  h2o.saveModel(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/")

data_transformed_tbl <- automl_models_h2o@leaderboard %>%
  as_tibble() %>%
  select(-c(rmse, mse)) %>% 
  mutate(model_type = str_extract(model_id, "[^_]+")) %>%
  slice(1:15) %>% 
  rownames_to_column(var = "rowname") %>%
  mutate(
    model_id   = as_factor(model_id) ,
    model_type = as.factor(model_type)
  ) %>% 
  pivot_longer(cols = -c(model_id, model_type, rowname), 
               names_to = "key", 
               values_to = "value", 
               names_transform = list(key = forcats::fct_inorder)
  ) 


data_transformed_tbl %>%
  ggplot(aes(value, model_id, color = model_type)) +
  geom_point(size = 3) +
  geom_label(aes(label = round(value, 2), hjust = "inward")) +
  
  labs(title = "Leaderboard Metrics",
       subtitle = paste0("Ordered by: ", "auc"),
       y = "Model Postion, Model ID", x = "") + 
  theme(legend.position = "bottom")




deeplearning_h2o <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_grid__1_AutoML_20201216_054400_model_1")

deeplearning_h2o

test_tbl

h2o.performance(deeplearning_h2o, newdata = as.h2o(test_tbl))


deeplearning_grid_01 <- h2o.grid(
  
  algorithm = "deeplearning",
  
  grid_id = "deeplearning_grid_01",
  
  x = x,
  y = y,
  
  training_frame   = train_h2o,
  validation_frame = valid_h2o,
  nfolds = 5,
  
  hyper_params = list(
    hidden = list(c(10, 10, 10), c(50, 20, 10), c(20, 20, 20)),
    epochs = c(10, 50, 100)
  )
)


deeplearning_grid_01


performance_h2o <- h2o.performance(deeplearning_h2o, newdata = as.h2o(test_tbl))

typeof(performance_h2o)
performance_h2o %>% slotNames()

performance_h2o@metrics


performance_tbl <- performance_h2o %>%
  h2o.metric() %>%
  as.tibble() 

performance_tbl %>% 
  glimpse()


theme_new <- theme(
  legend.position  = "bottom",
  legend.key       = element_blank(),,
  panel.background = element_rect(fill   = "transparent"),
  panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
  panel.grid.major = element_line(color = "grey", size = 0.333)
) 


performance_tbl %>%
  filter(f1 == max(f1))

performance_tbl %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = precision), color = "blue", size = 1) +
  geom_line(aes(y = recall), color = "red", size = 1) +
  
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) +
  labs(title = "Precision vs Recall", y = "value") +
  theme_new




path <- "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_grid__1_AutoML_20201216_054400_model_1"

load_model_performance_metrics <- function(path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
  
  perf_h2o %>%
    h2o.metric() %>%
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc)
  
}

load_model_performance_metrics <- function(path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
  
  perf_h2o %>%
    h2o.metric() %>%
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc)
  
}

model_metrics_tbl <- fs::dir_info(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/") %>%
  select(path) %>%
  mutate(metric = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = metric)


model_metrics_tbl %>%
  mutate(
    path = str_split(path, pattern = "/", simplify = T)[,3] %>% as_factor(),
    auc  = auc %>% round(3) %>% as.character() %>% as_factor()
  ) %>%
  ggplot(aes(fpr, tpr, color = path, linetype = auc)) +
  geom_line(size = 1) +
  
  geom_abline(color = "red", linetype = "dotted") +
  
  theme_new +
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "ROC Plot",
    subtitle = "Performance of 3 Top Performing Models"
  )


load_model_performance_metrics <- function(path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
  
  perf_h2o %>%
    h2o.metric() %>%
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc, precision, recall)
  
}

model_metrics_tbl <- fs::dir_info(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h2o_models/") %>%
  select(path) %>%
  mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = metrics)

model_metrics_tbl %>%
  mutate(
    path = str_split(path, pattern = "/", simplify = T)[,3] %>% as_factor(),
    auc  = auc %>% round(3) %>% as.character() %>% as_factor()
  ) %>%
  ggplot(aes(recall, precision, color = path, linetype = auc)) +
  geom_line(size = 1) +
  theme_new + 
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "Precision vs Recall Plot",
    subtitle = "Performance of 3 Top Performing Models"
  )


h2o_leaderboard <- automl_models_h2o@leaderboard
newdata <- test_tbl
order_by <- "auc"
max_models <- 4
size <- 1
  
leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() %>%
    slice(1:max_models)


gain_lift_tbl <- leaderboard_tbl %>%
  mutate(metrics = map(model_id, get_gain_lift, newdata_tbl)) %>%
  unnest(cols = metrics) %>%
  mutate(
    model_id = as_factor(model_id) %>% 
      fct_reorder(!! order_by_expr, 
                  .desc = ifelse(order_by == "auc", TRUE, FALSE)),
    auc  = auc %>% 
      round(3) %>% 
      as.character() %>% 
      as_factor() %>% 
      fct_reorder(as.numeric(model_id)),
    logloss = logloss %>% 
      round(4) %>% 
      as.character() %>% 
      as_factor() %>% 
      fct_reorder(as.numeric(model_id))
  ) %>%
  rename(
    gain = cumulative_capture_rate,
    lift = cumulative_lift
  ) 

gain_p <- gain_lift_tbl %>%
  ggplot(aes(cumulative_data_fraction, gain, 
             color = model_id, linetype = !! order_by_expr)) +
  geom_line(size = size,) +
  geom_segment(x = 0, y = 0, xend = 1, yend = 1, 
               color = "red", size = size, linetype = "dotted") +
  theme_new +
  expand_limits(x = c(0, 1), y = c(0, 1)) +
  labs(title = "Gain",
       x = "Cumulative Data Fraction", y = "Gain") +
  theme(legend.position = "none")

lift_p <- gain_lift_tbl %>%
  ggplot(aes(cumulative_data_fraction, lift, 
             color = model_id, linetype = !! order_by_expr)) +
  geom_line(size = size) +
  geom_segment(x = 0, y = 1, xend = 1, yend = 1, 
               color = "red", size = size, linetype = "dotted") +
  theme_new +
  expand_limits(x = c(0, 1), y = c(0, 1)) +
  labs(title = "Lift",
       x = "Cumulative Data Fraction", y = "Lift") +
  theme(legend.position = "none") 


p <- cowplot::plot_grid(gain_p, lift_p, ncol = 2)