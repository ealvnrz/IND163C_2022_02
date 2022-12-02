library(tidyverse)
library(tidymodels)
tidymodels_prefer()

# Pipeline ----

small_mtcars <- arrange(mtcars, gear)
small_mtcars <- slice(small_mtcars, 1:10)

# o de manera más compacta: 

small_mtcars <- slice(arrange(mtcars, gear), 1:10)

small_mtcars <- mtcars %>% 
  arrange(gear) %>% 
  slice(1:10)

# Desde R 4.1:

small_mtcars <-   mtcars |> 
  arrange(gear) |> 
  slice(1:10)

# Datos ----

ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))

## Semilla ----

set.seed(501)

## Split ----

ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_split

## Para guardar cada conjunto en objetos distintos

ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# Workflow ----

# objeto parsnip
lm_model <-  
  linear_reg() %>%  
  set_engine("lm")

# Workflow

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model)

lm_wflow <- 
  lm_wflow %>% 
  add_formula(Sale_Price ~ Longitude + Latitude)

lm_wflow

lm_fit <- fit(lm_wflow, ames_train)
lm_fit

predict(lm_fit, ames_test %>% slice(1:3))

# Agregar variables

lm_wflow <- 
  lm_wflow %>% 
  remove_formula() %>% 
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))
lm_wflow

fit(lm_wflow, ames_train)

# De manera resumida

lm_wflow <- 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))
lm_fit <- fit(lm_wflow, ames_train)


# Recipes ----

# Supongamos que queremos hacer este modelo lineal

# lm(Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Year_Built + Bldg_Type, data = ames)

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_dummy(all_nominal_predictors())
simple_ames

# Agregamos nuestro modelo al Workflow

# lm_wflow %>% 
#   add_recipe(simple_ames) # No se puede porque sólo se puede tener un método de procesamiento a la vez

# Borramos lo que teniamos antes

lm_wflow <- 
  lm_wflow %>% 
  remove_variables() %>% 
  add_recipe(simple_ames)
lm_wflow

# Estimamos

lm_fit <- fit(lm_wflow, ames_train)
predict(lm_fit, ames_test %>% slice(1:3))

# Obtener la recipe después de haber estimado

lm_fit %>% 
  extract_recipe(estimated = TRUE)

# Presentación de modelo

lm_fit %>% 
  extract_fit_parsnip() %>%  # Esto nos entrega el objeto de parsnip 
  tidy() %>% # Lo presentamos de manera bonita
  slice(1:5)

# Otros pasos del la receta/recipe

simple_ames <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
       data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors())

