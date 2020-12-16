library(tidymodels)
library(magrittr)
library(dplyr)
library(sjmisc)
library(magrittr)
library(haven)
library(sjlabelled)
employee_attrition_tbl <- read.csv("HR-Employee-Attrition.csv")

dept_job_role_tbl <- employee_attrition_tbl %>%
  select(EmployeeNumber, Department, JobRole, PerformanceRating, Attrition)

dept_job_role_tbl %>%
  
  group_by(Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  mutate(pct = n / sum(n))

dept_job_role_tbl %>%
  
  # Block 1
  group_by(Department, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  # Block 2: Caution: It's easy to inadvertently miss grouping when creating counts & percents within groups
  group_by(Department) %>%
  mutate(pct = n / sum(n))

dept_job_role_tbl %>%
  
  # Block 1
  group_by(Department, JobRole, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  # Block 2
  group_by(Department, JobRole) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  
  # Block 3
  filter(Attrition %in% "Yes")


###
dept_job_role_tbl %>%
  
  # Block 1
  group_by(Department, JobRole, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  # Block 2
  group_by(Department, JobRole) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  
  # Block 3
  filter(Attrition %in% "Yes") %>%
  arrange(desc(pct)) %>%
  mutate(
    above_industry_avg = case_when(
      pct > 0.088 ~ "Yes",
      TRUE ~ "No"
    )
  )


# Function to calculate attrition cost
calculate_attrition_cost <- function(
  
  # Employee
  n                    = 1,
  salary               = 80000,
  
  # Direct Costs
  separation_cost      = 500,
  vacancy_cost         = 10000,
  acquisition_cost     = 4900,
  placement_cost       = 3500,
  
  # Productivity Costs
  net_revenue_per_employee = 250000,
  workdays_per_year        = 240,
  workdays_position_open   = 40,
  workdays_onboarding      = 60,
  onboarding_efficiency    = 0.50
  
) {
  
  # Direct Costs
  direct_cost <- sum(separation_cost, vacancy_cost, acquisition_cost, placement_cost)
  
  # Lost Productivity Costs
  productivity_cost <- net_revenue_per_employee / workdays_per_year *
    (workdays_position_open + workdays_onboarding * onboarding_efficiency)
  
  # Savings of Salary & Benefits (Cost Reduction)
  salary_benefit_reduction <- salary / workdays_per_year * workdays_position_open
  
  # Estimated Turnover Per Employee
  cost_per_employee <- direct_cost + productivity_cost - salary_benefit_reduction
  
  # Total Cost of Employee Turnover
  total_cost <- n * cost_per_employee
  
  return(total_cost)
  
}

calculate_attrition_cost()
## [1] 78483.33
calculate_attrition_cost(200)
## [1] 15696667


dept_job_role_tbl %>%
  
  # Block 1
  group_by(Department, JobRole, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  # Block 2
  group_by(Department, JobRole) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  
  # Block 3
  filter(Attrition %in% "Yes") %>%
  arrange(desc(pct)) %>%
  mutate(
    above_industry_avg = case_when(
      pct > 0.088 ~ "Yes",
      TRUE ~ "No"
    )
  ) %>%
  
  # Block 4. Set salaray to 80000 for now
  mutate(
    cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
  )

dept_job_role_tbl %>%
  count(Department, JobRole, Attrition)


dept_job_role_tbl %>%
  
  group_by(Department, JobRole, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  group_by(Department, JobRole) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  
  filter(Attrition %in% "Yes") %>%
  arrange(desc(pct)) %>%
  mutate(
    above_industry_avg = case_when(
      pct > 0.088 ~ "Yes",
      TRUE ~ "No"
    )
  ) %>%
  
  mutate(
    cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
  )



assess_attrition <- function(data, attrition_col, attrition_value, baseline_pct) {
  
  attrition_col_expr <- enquo(attrition_col)
  
  data %>%
    
    # Use parenthesis () to give tidy eval evaluation priority
    filter((!! attrition_col_expr) %in% attrition_value) %>%
    arrange(desc(pct)) %>%
    mutate(
      # Function inputs in numeric format (e.g. baseline_pct = 0.088 don't require tidy eval)
      above_industry_avg = case_when(
        pct > baseline_pct ~ "Yes",
        TRUE ~ "No"
      )
    )
  
}

dept_job_role_tbl %>%
  
  count(Department, JobRole, Attrition) %>%
  count_to_pct(Department, JobRole) %>%
  assess_attrition(Attrition, attrition_value = "Yes", baseline_pct = 0.088) %>%
  mutate(
    cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
  ) %>%
  
  # Data Manipulation
  mutate(name = str_c(Department, JobRole, sep = ": ") %>% as_factor()) %>%
  
  # Check levels
  # pull(name) %>%
  # levels()
  
  mutate(name      = fct_reorder(name, cost_of_attrition)) %>%
  mutate(cost_text = str_c("$", format(cost_of_attrition / 1e6, digits = 2),
                           "M", sep = "")) %>%
  
  #Plotting
  ggplot(aes(cost_of_attrition, y = name)) +
  geom_segment(aes(xend = 0, yend = name),    color = "#2dc6d6") +
  geom_point(  aes(size = cost_of_attrition), color = "#2dc6d6") +
  scale_x_continuous(labels = scales::dollar) +
  geom_label(aes(label = cost_text, size = cost_of_attrition),
             hjust = "inward", color = "#2dc6d6") +
  scale_size(range = c(3, 5)) +
  labs(title = "Estimated cost of Attrition: By Dept and Job Role",
       y = "",
       x = "Cost of attrition") +
  theme(legend.position = "none")

dept_job_role_tbl %>%
  
  group_by(Department, JobRole, Attrition) %>%
  summarize(n = n()) %>%
  ungroup() %>%
  
  group_by(Department, JobRole) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  
  filter(Attrition %in% "Yes") %>%
  arrange(desc(pct)) %>%
  mutate(
    above_industry_avg = case_when(
      pct > 0.088 ~ "Yes",
      TRUE ~ "No"
    )
  ) %>%
  
  mutate(
    cost_of_attrition = calculate_attrition_cost(n = n, salary = 80000)
  )



# Descriptive Features
employee_attrition_tbl %>% select(Age, DistanceFromHome, Gender, MaritalStatus, NumCompaniesWorked, Over18)

# Employment Features
employee_attrition_tbl %>% select(Department, EmployeeCount, EmployeeNumber, JobInvolvement, JobLevel, JobRole, JobSatisfaction)

# Compensation Features
employee_attrition_tbl %>% select(DailyRate, HourlyRate, MonthlyIncome, MonthlyRate, PercentSalaryHike, StockOptionLevel)

# Survery Results
employee_attrition_tbl %>% select(EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance)

# Performance Data
employee_attrition_tbl %>% select(JobInvolvement, PerformanceRating)

# Work-Life Features
employee_attrition_tbl %>% select(BusinessTravel, OverTime)

# Training & Education
employee_attrition_tbl %>% select(Education, EducationField, TrainingTimesLastYear)

# Time-Based Features
employee_attrition_tbl %>% select(TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager)


library(GGally)
# Step 2: Data Visualization ----

employee_attrition_tbl %>%
  select(Attrition,  Gender, MaritalStatus, NumCompaniesWorked, Over18, DistanceFromHome) %>%
  ggpairs()


# Explore Features by Category

plot_ggpairs <- function(data, color = NULL, density_alpha = 0.5) {
  
  color_expr <- enquo(color)
  
  if (rlang::quo_is_null(color_expr)) {
    
    g <- data %>%
      ggpairs(lower = "blank") 
    
  } else {
    
    color_name <- quo_name(color_expr)
    
    g <- data %>%
      ggpairs(mapping = aes_string(color = color_name), 
              lower = "blank", legend = 1,
              diag = list(continuous = wrap("densityDiag", 
                                            alpha = density_alpha))) +
      theme(legend.position = "bottom")
  }
  
  return(g)
  
}

#   1. Descriptive features: age, gender, marital status 
employee_attrition_tbl %>%
  select(Attrition, Gender, MaritalStatus, NumCompaniesWorked, Over18, DistanceFromHome) %>%
  plot_ggpairs(Attrition)

#   2. Employment features: department, job role, job level
employee_attrition_tbl %>%
  select(Attrition, contains("employee"), contains("department"), contains("job")) %>%
  plot_ggpairs(Attrition) 

#   3. Compensation features: HourlyRate, MonthlyIncome, StockOptionLevel 
employee_attrition_tbl %>%
  select(Attrition, contains("income"), contains("rate"), contains("salary"), contains("stock")) %>%
  plot_ggpairs(Attrition)

#   4. Survey Results: Satisfaction level, WorkLifeBalance 
employee_attrition_tbl %>%
  select(Attrition, contains("satisfaction"), contains("life")) %>%
  plot_ggpairs(Attrition)

#   5. Performance Data: Job Involvment, Performance Rating
employee_attrition_tbl %>%
  select(Attrition, contains("performance"), contains("involvement")) %>%
  plot_ggpairs(Attrition)

#   6. Work-Life Features 
employee_attrition_tbl %>%
  select(Attrition, contains("overtime"), contains("travel")) %>%
  plot_ggpairs(Attrition)

#   7. Training and Education 
employee_attrition_tbl %>%
  select(Attrition, contains("training"), contains("education")) %>%
  plot_ggpairs(Attrition)

#   8. Time-Based Features: Years at company, years in current role
employee_attrition_tbl %>%
  select(Attrition, contains("years")) %>%
  plot_ggpairs(Attrition)
