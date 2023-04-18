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
library(stringr)

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






automl_models_h2o@leaderboard %>% 
  as_tibble() %>% 
  select(-c(mean_per_class_error, rmse, mse))




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




deeplearning_h2o <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/DRF_1_AutoML_20201224_182023")

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
h2o.getGrid(grid_id = "deeplearning_grid_01", sort_by = "auc", decreasing = TRUE)
deeplearning_grid_01_model_1 <- h2o.getModel("deeplearning_grid_01_model_1")

deeplearning_grid_01_model_1 %>% h2o.auc(train = T, valid = T, xval = T)

deeplearning_grid_01_model_1 %>%
  h2o.performance(newdata = as.h2o(test_tbl))

# deeplearning_grid_01_model_1 %>%h2o.saveModel(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/")



stacked_ensemble_h2o <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_grid__1_AutoML_20201224_184436_model_2")
deeplearning_h2o     <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/DRF_1_AutoML_20201224_182023")
glm_h2o              <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/deeplearning_grid_01_model_1")


performance_h2o <- h2o.performance(stacked_ensemble_h2o, newdata = as.h2o(test_tbl))

typeof(performance_h2o)
performance_h2o %>% slotNames()

performance_h2o@metrics



h2o.auc(performance_h2o, train = T, valid = T, xval = T)

h2o.auc(stacked_ensemble_h2o, train = T, valid = T, xval = T)

h2o.giniCoef(performance_h2o)
h2o.logloss(performance_h2o)
h2o.confusionMatrix(stacked_ensemble_h2o)
h2o.confusionMatrix(performance_h2o)

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
  
  # Insert line where precision and recall are harmonically optimized
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) +
  labs(title = "Precision vs Recall", y = "value") +
  theme_new




# ROC Plot

path <- "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_grid__1_AutoML_20201224_184436_model_2"

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
  mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = metrics)

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




# Precision vs Recall

load_model_performance_metrics <- function(path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
  
  perf_h2o %>%
    h2o.metric() %>%
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc, precision, recall)
  
}

model_metrics_tbl <- fs::dir_info(path = "C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/") %>%
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

stacked_ensemble_h2o <- h2o.loadModel("C:/Users/sagor/Documents/GitHub/ml_journal-ttsagor/h20_models/GBM_1_AutoML_20201224_205832")
stacked_ensemble_h2o

predictions <- h2o.predict(stacked_ensemble_h2o, newdata = as.h2o(test_tbl))

typeof(predictions)

predictions_tbl <- predictions %>% as_tibble()
predictions_tbl


# Gain & Lift
ranked_predictions_tbl <- predictions_tbl %>%
  bind_cols(test_tbl) %>%
  select(predict:Yes, went_on_backorder) %>%
  arrange(desc(Yes))


## Gain Chart
ranked_predictions_tbl %>%
  mutate(ntile = ntile(Yes, n = 10)) %>%
  group_by(ntile) %>%
  summarise(
    cases = n(),
    responses = sum(went_on_backorder == "Yes")
  ) %>%
  arrange(desc(ntile))
calculated_gain_lift_tbl <- ranked_predictions_tbl %>%
  mutate(ntile = ntile(Yes, n = 10)) %>%
  group_by(ntile) %>%
  summarise(
    cases = n(),
    responses = sum(went_on_backorder == "Yes")
  ) %>%
  arrange(desc(ntile)) %>%
  
  # Add group numbers (opposite of ntile)
  mutate(group = row_number()) %>%
  select(group, cases, responses) %>%
  
  # Calculations
  mutate(
    cumulative_responses = cumsum(responses),
    pct_responses        = responses / sum(responses),
    gain                 = cumsum(pct_responses),
    cumulative_pct_cases = cumsum(cases) / sum(cases),
    lift                 = gain / cumulative_pct_cases,
    gain_baseline        = cumulative_pct_cases,
    lift_baseline        = gain_baseline / cumulative_pct_cases
  )

calculated_gain_lift_tbl 
gain_lift_tbl <- performance_h2o %>%
  h2o.gainsLift() %>%
  as.tibble()
gain_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("lift")) %>%
  mutate(baseline = cumulative_data_fraction) %>%
  dplyr::rename(gain     = cumulative_capture_rate) %>%
  # prepare the data for the plotting (for the color and group aesthetics)
  pivot_longer(cols = c(gain, baseline), values_to = "value", names_to = "key")

gain_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Gain Chart",
    x = "Cumulative Data Fraction",
    y = "Gain"
  ) +
  theme_new

## Lift Plot

lift_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("capture")) %>%
  mutate(baseline = 1) %>%
  dplyr::rename(lift = cumulative_lift) %>%
  pivot_longer(cols = c(lift, baseline), values_to = "value", names_to = "key")

lift_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Lift Chart",
    x = "Cumulative Data Fraction",
    y = "Lift"
  ) +
  theme_new


# cow plot
h2o_leaderboard <- automl_models_h2o@leaderboard
newdata <- test_tbl
order_by <- "auc"
max_models <- 4
size <- 1

plot_h2o_performance <- function(h2o_leaderboard, newdata, order_by = c("auc", "logloss"),
                                 max_models = 3, size = 1.5) {
  
  
  leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() 
  
  newdata_tbl <- newdata %>%
    as_tibble()
  
  order_by      <- tolower(order_by[[1]]) 
  
  order_by_expr <- rlang::sym(order_by)
  
  h2o.no_progress()
  
  get_model_performance_metrics <- function(model_id, test_tbl) {
    
    model_h2o <- h2o.getModel(model_id)
    perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    
    perf_h2o %>%
      h2o.metric() %>%
      as.tibble() %>%
      select(threshold, tpr, fpr, precision, recall)
    
  }
  
  model_metrics_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_model_performance_metrics, newdata_tbl)) %>%
    unnest(cols = metrics) %>%
    mutate(
      model_id = as_factor(model_id), 
      
      auc      = auc %>% 
        round(3) %>% 
        as.character() %>% 
        as_factor(),
      
      logloss  = logloss %>% 
        round(4) %>% 
        as.character() %>% 
        as_factor() 
      
    )
  
  
  # 1A. ROC Plot
  
  p1 <- model_metrics_tbl %>%
    ggplot(aes(fpr, tpr, color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    theme_new +
    labs(title = "ROC", x = "FPR", y = "TPR") +
    theme(legend.direction = "vertical") 
  
  
  # 1B. Precision vs Recall
  
  p2 <- model_metrics_tbl %>%
    ggplot(aes(recall, precision, color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    theme_new +
    labs(title = "Precision Vs Recall", x = "Recall", y = "Precision") +
    theme(legend.position = "none") 
  
  
  # 2. Gain / Lift
  
  get_gain_lift <- function(model_id, test_tbl) {
    
    model_h2o <- h2o.getModel(model_id)
    perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
    
    perf_h2o %>%
      h2o.gainsLift() %>%
      as.tibble() %>%
      select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift)
    
  }
  
  gain_lift_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_gain_lift, newdata_tbl)) %>%
    unnest(cols = metrics) %>%
    mutate(
      model_id = as_factor(model_id),
      auc  = auc %>% 
        round(3) %>% 
        as.character() %>% 
        as_factor(),
      logloss = logloss %>% 
        round(4) %>% 
        as.character() %>% 
        as_factor()
    ) %>%
    dplyr::rename(
      gain = cumulative_capture_rate,
      lift = cumulative_lift
    ) 
  
  # 2A. Gain Plot
  
  p3 <- gain_lift_tbl %>%
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
  
  # 2B. Lift Plot
  
  p4 <- gain_lift_tbl %>%
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
  
  
  # Combine using cowplot
  
  # cowplot::get_legend extracts a legend from a ggplot object
  p_legend <- get_legend(p1)
  # Remove legend from p1
  p1 <- p1 + theme(legend.position = "none")
  
  # cowplot::plt_grid() combines multiple ggplots into a single cowplot object
  p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)
  
  # cowplot::ggdraw() sets up a drawing layer
  p_title <- ggdraw() + 
    
    # cowplot::draw_label() draws text on a ggdraw layer / ggplot object
    draw_label("H2O Model Metrics", size = 18, fontface = "bold", 
               color = "#2C3E50")
  
  p_subtitle <- ggdraw() + 
    draw_label(glue("Ordered by {toupper(order_by)}"), size = 10,  
               color = "#2C3E50")
  
  # Combine everything
  ret <- plot_grid(p_title, p_subtitle, p, p_legend, 
                   
                   # Adjust the relative spacing, so that the legends always fits
                   ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.05 * max_models))
  
  h2o.show_progress()
  
  return(ret)
  
}

automl_models_h2o@leaderboard %>%
  plot_h2o_performance(newdata = test_tbl, order_by = "logloss", 
                       size = 0.5, max_models = 4)