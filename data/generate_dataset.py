"""
Realistic Enterprise HR Attrition Dataset Generator.
Synthesizes 15,000 enterprise employee records reflecting empirical HR turnover dynamics.
"""

import numpy as np
import pandas as pd
import os

def generate_hr_dataset(n_samples=15000, random_state=42):
    np.random.seed(random_state)
    
    departments = ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management']
    dept_weights = [0.27, 0.18, 0.15, 0.08, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04]
    
    salaries = ['low', 'medium', 'high']
    salary_weights = [0.48, 0.43, 0.09]
    
    dept_col = np.random.choice(departments, size=n_samples, p=dept_weights)
    salary_col = np.random.choice(salaries, size=n_samples, p=salary_weights)
    
    # Base distributions
    satisfaction = np.random.beta(a=5, b=3, size=n_samples) # mean ~0.62
    last_evaluation = np.random.beta(a=6, b=3, size=n_samples) # mean ~0.66
    number_projects = np.random.choice([2, 3, 4, 5, 6, 7], size=n_samples, p=[0.16, 0.27, 0.30, 0.18, 0.06, 0.03])
    avg_monthly_hours = np.random.normal(loc=200, scale=45, size=n_samples).astype(int)
    avg_monthly_hours = np.clip(avg_monthly_hours, 96, 315)
    
    time_spent_company = np.random.choice([2, 3, 4, 5, 6, 7, 8, 10], size=n_samples, p=[0.21, 0.43, 0.17, 0.10, 0.05, 0.02, 0.01, 0.01])
    work_accident = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    promotion_last_5years = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    overtime = np.random.choice([0, 1], size=n_samples, p=[0.68, 0.32])
    commute_distance_km = np.random.randint(1, 55, size=n_samples)
    
    # Calculate empirical turnover probability based on HR behavioral archetypes
    turnover_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.08 # baseline background attrition (~8%)
        
        # Archetype 1: Overworked High Performer Burnout
        if last_evaluation[i] > 0.78 and avg_monthly_hours[i] > 240 and number_projects[i] >= 5:
            prob += 0.65
            if salary_col[i] == 'low':
                prob += 0.15
        
        # Archetype 2: Under-engaged / Disgruntled
        if satisfaction[i] < 0.38 and avg_monthly_hours[i] < 160 and number_projects[i] <= 3:
            prob += 0.60
            
        # Archetype 3: The 3-5 Year Career Plateau / Stagnation
        if time_spent_company[i] in [3, 4, 5]:
            if promotion_last_5years[i] == 0:
                prob += 0.18
            if satisfaction[i] < 0.50:
                prob += 0.25
                
        # Archetype 4: Overtime + Long Commute + Low Salary combo
        if overtime[i] == 1 and commute_distance_km[i] > 30 and salary_col[i] == 'low':
            prob += 0.22
            
        # Mitigating retention factors
        if salary_col[i] == 'high':
            prob -= 0.15
        if promotion_last_5years[i] == 1:
            prob -= 0.20
        if satisfaction[i] > 0.80 and avg_monthly_hours[i] < 230:
            prob -= 0.25
        if time_spent_company[i] >= 7: # Long-tenure loyalty
            prob -= 0.12
            
        turnover_prob[i] = np.clip(prob, 0.02, 0.96)
        
    turnover = (np.random.rand(n_samples) < turnover_prob).astype(int)
    
    df = pd.DataFrame({
        'satisfaction_level': np.round(satisfaction, 2),
        'last_evaluation': np.round(last_evaluation, 2),
        'number_projects': number_projects,
        'average_monthly_hours': avg_monthly_hours,
        'time_spent_company': time_spent_company,
        'work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'department': dept_col,
        'salary_level': salary_col,
        'overtime': overtime,
        'commute_distance_km': commute_distance_km,
        'turnover': turnover
    })
    
    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Generating 15,000 enterprise employee records...")
    df = generate_hr_dataset(15000, random_state=42)
    
    # Train/Test Split (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    train_path = os.path.join(script_dir, "hr_attrition_train.csv")
    test_path = os.path.join(script_dir, "hr_attrition_test.csv")
    sample_batch_path = os.path.join(script_dir, "sample_batch_employees.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Create sample batch of 25 records (without turnover column) for testing the UI
    sample_batch = test_df.drop(columns=['turnover']).head(25).copy()
    sample_batch.insert(0, 'employee_id', [f"EMP-{1000+i}" for i in range(25)])
    sample_batch.to_csv(sample_batch_path, index=False)
    
    turnover_rate = df['turnover'].mean() * 100
    print(f"Generated successfully!")
    print(f"  Total records: {len(df):,}")
    print(f"  Training set:  {len(train_df):,} -> {train_path}")
    print(f"  Testing set:   {len(test_df):,} -> {test_path}")
    print(f"  Sample batch:  25 records -> {sample_batch_path}")
    print(f"  Overall Turnover Rate: {turnover_rate:.1f}%")

if __name__ == '__main__':
    main()
