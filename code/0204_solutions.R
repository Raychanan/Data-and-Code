# In class code - Thursday - Modeling

# solutions

# load some libraries
library(tidyverse) # pipe, ggplot, etc.
library(broom) # augment function
library(rcfss) # logit2prob function
library(tidymodels) # accuracy function
library(here) 

# 1. Load the titanic data (`titanic.csv`), which has been cleaned and preprocessed for you including only relevant features for this challenge. 
titanic <- read_csv(here("data", "titanic.csv"))

titanic$Survived <- as.factor(titanic$Survived) # ensure it's a factor variable

# 2. Create a train/test split, at 70/30 respectively.
set.seed(234)

titanic_split <- initial_split(titanic, prop = 0.7)
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

# 3. Fit *100* versions of a kNN classifier to the training data, with each fit corresponding to a different value of k, from 1 to 100. 
# 4. Evaluate *each* kNN classifier using the testing (30%) set. Be sure to store the results for each fit. 
mse_knn <- tibble(k = 1:100,
                  knn_train = map(k, ~ class::knn(dplyr::select(titanic_train, -Survived),
                                                  test = dplyr::select(titanic_train, -Survived),
                                                  cl = titanic_train$Survived, k = .)),
                  knn_test = map(k, ~ class::knn(dplyr::select(titanic_train, -Survived),
                                                 test = dplyr::select(titanic_test, -Survived),
                                                 cl = titanic_train$Survived, k = .)),
                  err_train = map_dbl(knn_train, ~ mean(titanic_test$Survived != .)),
                  err_test = map_dbl(knn_test, ~ mean(titanic_test$Survived != .)))

# 5. Fit a logistic regression to the training data, predicting the probability of survival (`Survived`) as a function of all other features in the data.
titanic_logit <- glm(Survived ~ ., data = titanic_train, family = binomial)

# 6. Evaluate the logistic regression using the testing (30%) set. Be sure to store the results. 
titanic_logit_error <- augment(titanic_logit, newdata = titanic_test) %>% 
  as_tibble() %>%
  mutate(.prob = logit2prob(.fitted),
         .pred = factor(round(.prob))) %>%
  accuracy(truth = Survived, estimate = .pred)

# 7. Plot the test error rate from all fits of the kNN classifier as line plot (with values of k ranging along the X axis), and place a horizontal reference line on the plot showing the test error from the logistic regression. 
ggplot(mse_knn, aes(k, err_test)) +
  geom_line() +
  geom_hline(yintercept = 1 - titanic_logit_error$.estimate[[1]], linetype = 2) +
  labs(x = "K",
       y = "Test error rate") +
  expand_limits(y = 0) + 
  theme_minimal()
