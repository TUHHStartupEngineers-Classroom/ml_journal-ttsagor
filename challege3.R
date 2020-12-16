library(tidymodels)
library(rstanarm)
library(broom.mixed)

bike_data_tbl <- readRDS("raw_data/bike_orderlines.rds")

ggplot(bike_data_tbl,
       aes(x = price, 
           y = weight, 
           group = category_1, 
           col = category_1)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE) +
  scale_color_manual(values=c("#2dc6d6", "#d65a2d", "#d6af2d", "#8a2dd6"))


data_split <- initial_split(bike_data_tbl, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

# recipe
bike_rec <- 
  recipe(price ~ ., data = train_data) %>% 
  update_role(url, new_role = "URL") %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  prep()

bike_rec

lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

lr_mod

bike_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(bike_rec)
bike_wflow


bike_fit <- 
  bike_wflow %>% 
  fit(data = train_data)

bike_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

predict(bike_fit, test_data)

bike_pred <- 
  predict(bike_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(price, category_2)) 

bike_pred %>% 
  roc_curve(truth = price, category_2) %>% 
  autoplot()


bike_pred %>% 
  roc_auc(truth = arr_delay, category_2)


model_01_linear_lm_simple <- linear_reg(mode = "regression") %>%
  set_engine("lm") %>%
  fit(price ~ category_2 + frame_material, data = train_data)


new_cross_country <- tibble(
  model = "Exceed AL SL new",
  category_2 = "Cross-Country",
  frame_material = "aluminium",
  shimano_dura_ace = 0,
  shimano_ultegra = 0,
  shimano_105 = 0,
  shimano_tiagra = 0,
  Shimano_sora = 0,
  shimano_deore = 0,
  shimano_slx = 0,
  shimano_grx = 0,
  Shimano_xt = 1,
  Shimano_xtr = 0,
  Shimano_saint = 0,
  SRAM_red = 0,
  SRAM_force = 0,
  SRAM_rival = 0,
  SRAM_apex = 0,
  SRAM_xx1 = 0,
  SRAM_x01 = 0,
  SRAM_gx = 0,
  SRAM_nx = 0,
  SRAM_sx = 0,
  Campagnolo_potenza = 0,
  Campagnolo_super_record = 0,
  shimano_nexus = 0,
  shimano_alfine = 0
) 

new_cross_country

set.seed(1234)
model_07_boost_tree_xgboost <- boost_tree(
  mode = "regression",
  mtry = 30,
  learn_rate = 0.25,
  tree_depth = 7
) %>%
  set_engine("xgboost") %>%
  fit(price ~ ., data = train_data %>% select(-c(id:weight), -category_1, -c(category_3:gender)))

model_07_boost_tree_xgboost %>% calc_metrics(test_data)

predict(model_01_linear_lm_simple, new_data = new_cross_country)
predict(model_07_boost_tree_xgboost, new_data = new_cross_country)

models_tbl <- tibble(
  model_id = str_c("Model 0", 1:2),
  model = list(
    model_01_linear_lm_simple,
    model_07_boost_tree_xgboost
  )
)

models_tbl

predictions_new_cross_country_tbl <- models_tbl %>%
  mutate(predictions = map(model, predict, new_data = new_cross_country)) %>%
  unnest(predictions) %>%
  mutate(category_2 = "Cross-Country") %>%
  left_join(new_cross_country, by = "category_2")

predictions_new_cross_country_tbl


g1 <- geom_point(aes(y = .pred), color = "red", alpha = 0.5,
             data = predictions_new_cross_country_tbl) +
  ggrepel::geom_text_repel(aes(label = model_id, y = .pred),
                           size = 3,
                           data = predictions_new_cross_country_tbl)
