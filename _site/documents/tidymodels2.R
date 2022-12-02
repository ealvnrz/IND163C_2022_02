########### Required Packages ###########

packages = c("dplyr", "bayesplot" ,"lme4", "RcppEigen", # "rstan","shinystan", 
             "tidyverse", "tidyr", "AmesHousing", "broom", "caret", "dials", "doParallel", "e1071", "earth",
             "ggrepel", "glmnet", "ipred", "klaR", "kknn", "pROC", "rpart", "randomForest", "tune",
             "sessioninfo", "tidymodels","ranger", "recipes", "workflows", "themis","xgboost")


lapply(packages, require, character.only = TRUE)


set.seed(717)

theme_set(theme_bw())

"%!in%" <- Negate("%in%")
g <- glimpse

### Data Preperation

### Set up Ames Housing Data

ames <- make_ames() %>% 
  dplyr::select(-matches("Qu")) %>% 
  filter(Neighborhood  %!in% c("Green_Hills", "Landmark", "Blueste", "Greens")) %>% 
  mutate(Neighborhood = as.character(Neighborhood)) %>% 
  dplyr::select(Sale_Price, Latitude, Longitude, Pool_QC, Paved_Drive,
                Garage_Area, Fireplaces, First_Flr_SF, Full_Bath, 
                Neighborhood, Lot_Area, Bldg_Type)

ames <- sample_n(ames, 1000)

### Initial Split for Training and Test

data_split <- rsample::initial_split(ames, strata = "Sale_Price", prop = 0.75)

ames_train <- training(data_split)
ames_test  <- testing(data_split)


### Cross Validation

# K-fold, K = 10

cv_splits_v5 <- vfold_cv(ames_train, v = 3, strata = "Sale_Price")
print(cv_splits_v5)

### Create Recipes

# Feature Creation

model_rec <- recipe(Sale_Price ~ ., data = ames_train) %>%
  update_role(Neighborhood, new_role = "Neighborhood") %>%
  step_other(Neighborhood, threshold = 0.005) %>%
  step_dummy(all_nominal(), -Neighborhood) %>%
  step_log(Sale_Price) %>% 
  step_zv(all_predictors()) %>%
  step_center(all_predictors(), -Sale_Price) %>%
  step_scale(all_predictors(), -Sale_Price) %>%
  step_ns(Latitude, Longitude, options = list(df = 4))

# See the data after all transformations

glimpse(model_rec %>% prep() %>% juice())


## Model specifications

lm_plan <- 
  linear_reg() %>% 
  set_engine("lm")

glmnet_plan <- 
  linear_reg() %>% 
  set_args(penalty  = tune()) %>%
  set_args(mixture  = tune()) %>%
  set_engine("glmnet")

rf_plan <- rand_forest() %>%
  set_args(mtry  = tune()) %>%
  set_args(min_n = tune()) %>%
  set_args(trees = 1000) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("regression")

XGB_plan <- boost_tree() %>%
  set_args(mtry  = tune()) %>%
  set_args(min_n = tune()) %>%
  set_args(trees = 100) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


# Hyperparameter grid for glmnet (penalization)

glmnet_grid <- expand.grid(penalty = seq(0, 1, by = .20), 
                           mixture = seq(0,1,0.1))
rf_grid <- expand.grid(mtry = c(2,5), 
                       min_n = c(1,5))
xgb_grid <- expand.grid(mtry = c(3,5), 
                        min_n = c(1,5))


# create workflow

lm_wf <-
  workflow() %>% 
  add_recipe(model_rec) %>% 
  add_model(lm_plan)

glmnet_wf <-
  workflow() %>% 
  add_recipe(model_rec) %>% 
  add_model(glmnet_plan)

rf_wf <-
  workflow() %>% 
  add_recipe(model_rec) %>% 
  add_model(rf_plan)

xgb_wf <-
  workflow() %>% 
  add_recipe(model_rec) %>% 
  add_model(XGB_plan)


# fit model to workflow and calculate metrics

control <- control_resamples(save_pred = TRUE, verbose = TRUE)

lm_tuned <- lm_wf %>%
  tune::fit_resamples(.,
                      resamples = cv_splits_v5,
                      control   = control,
                      metrics   = metric_set(rmse, rsq))

glmnet_tuned <- glmnet_wf %>%
  tune::tune_grid(.,
                  resamples = cv_splits_v5,
                  grid      = glmnet_grid,
                  control   = control,
                  metrics   = metric_set(rmse, rsq))

rf_tuned <- rf_wf %>%
  tune::tune_grid(.,
                  resamples = cv_splits_v5,
                  grid      = rf_grid,
                  control   = control,
                  metrics   = metric_set(rmse, rsq))

xgb_tuned <- xgb_wf %>%
  tune::tune_grid(.,
                  resamples = cv_splits_v5,
                  grid      = xgb_grid,
                  control   = control,
                  metrics   = metric_set(rmse, rsq))


## metrics across grid

autoplot(xgb_tuned)
collect_metrics(xgb_tuned)

## 'Best' by some metric and margin

show_best(lm_tuned, metric = "rmse", n = 15, maximize = FALSE)
show_best(glmnet_tuned, metric = "rmse", n = 15, maximize = FALSE)
show_best(rf_tuned, metric = "rmse", n = 15, maximize = FALSE)
show_best(xgb_tuned, metric = "rmse", n = 15, maximize = FALSE)

lm_best_params     <- select_best(lm_tuned, metric = "rmse", maximize = FALSE)
glmnet_best_params <- select_best(glmnet_tuned, metric = "rmse", maximize = FALSE)
rf_best_params     <- select_best(rf_tuned, metric = "rmse", maximize = FALSE)
xgb_best_params    <- select_best(xgb_tuned, metric = "rmse", maximize = FALSE)

## Final workflow

lm_best_wf     <- finalize_workflow(lm_wf, lm_best_params)
glmnet_best_wf <- finalize_workflow(glmnet_wf, glmnet_best_params)
rf_best_wf     <- finalize_workflow(rf_wf, rf_best_params)
xgb_best_wf    <- finalize_workflow(xgb_wf, xgb_best_params)


### Fold Predictions for best param set 


lm_OOF_preds     <- collect_predictions(lm_tuned) 
glmnet_OOF_preds <- collect_predictions(glmnet_tuned) %>% 
  filter(penalty == glmnet_best_params$penalty[1],
         mixture == glmnet_best_params$mixture[1])
rf_OOF_preds     <- collect_predictions(rf_tuned) %>% 
  filter(mtry  == rf_best_params$mtry[1],
         min_n == rf_best_params$min_n[1])
xgb_OOF_preds    <- collect_predictions(xgb_tuned) %>% 
  filter(mtry  == xgb_best_params$mtry[1],
         min_n == xgb_best_params$min_n[1])


# Visualize pred and matrix of hyperparameters

ggplot(collect_metrics(glmnet_tuned) %>% filter(.metric == "rmse"),
       aes(x=factor(penalty),y=factor(mixture),fill=mean)) + 
  geom_raster() +
  theme_minimal() +
  coord_fixed() +
  labs(x="Penalty",y="Mixture",title="Hyperparameter Tune Grid Results") +
  scale_fill_viridis_c(option = "A") +
  facet_wrap(~.metric)
ggplot(collect_metrics(glmnet_tuned) %>% filter(.metric == "rsq"),
       aes(x=factor(penalty),y=factor(mixture),fill=mean)) + 
  geom_raster() +
  theme_minimal() +
  coord_fixed() +
  labs(x="Penalty",y="Mixture",title="Hyperparameter Tune Grid Results") +
  scale_fill_viridis_c(option = "A") +
  facet_wrap(~.metric)
ggplot(collect_metrics(rf_tuned) %>% filter(.metric == "rmse"),
       aes(x=factor(mtry),y=factor(min_n),fill=mean)) + 
  geom_raster() +
  theme_minimal() +
  coord_fixed() +
  labs(x="mtry",y="min.node.size",title="Hyperparameter Tune Grid Results") +
  scale_fill_viridis_c(option = "A") +
  facet_wrap(~.metric)
ggplot(collect_metrics(xgb_tuned) %>% filter(.metric == "rmse"),
       aes(x=factor(mtry),y=factor(min_n),fill=mean)) + 
  geom_raster() +
  theme_minimal() +
  coord_fixed() +
  labs(x="mtry",y="min.node.size",title="Hyperparameter Tune Grid Results") +
  scale_fill_viridis_c(option = "A") +
  facet_wrap(~.metric)

# Fit

lm_val_fit <- lm_best_wf %>% 
  last_fit(split     = data_split,
           control   = control,
           metrics   = metric_set(rmse, rsq))

glmnet_val_fit <- glmnet_best_wf %>% 
  last_fit(split     = data_split,
           control   = control,
           metrics   = metric_set(rmse, rsq))

rf_val_fit <- rf_best_wf %>% 
  last_fit(split     = data_split,
           control   = control,
           metrics   = metric_set(rmse, rsq))

xgb_val_fit <- xgb_best_wf %>% 
  last_fit(split     = data_split,
           control   = control,
           metrics   = metric_set(rmse, rsq))

# collect test set predictions

lm_val_pred     <- collect_predictions(lm_val_fit)
glmnet_val_pred <- collect_predictions(glmnet_val_fit)
rf_val_pred     <- collect_predictions(rf_val_fit)
xgb_val_pred    <- collect_predictions(xgb_val_fit)

# show test set metrics

collect_metrics(lm_val_fit)
collect_metrics(glmnet_val_fit)
collect_metrics(rf_val_fit)
collect_metrics(xgb_val_fit)

