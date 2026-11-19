"""
Model Explainability & Actionable Retention Recommendations Engine.
Identifies primary risk drivers for individual employees and produces tailored HR retention strategies.
"""

def extract_employee_risk_factors(employee_dict):
    """
    Identifies specific risk drivers contributing to turnover probability.
    Returns list of risk driver objects with severity, factor name, and rationale.
    """
    risk_factors = []
    
    sat = float(employee_dict.get('satisfaction_level', 0.5))
    eval_score = float(employee_dict.get('last_evaluation', 0.5))
    hours = int(employee_dict.get('average_monthly_hours', 200))
    projects = int(employee_dict.get('number_projects', 3))
    tenure = int(employee_dict.get('time_spent_company', 3))
    promoted = int(employee_dict.get('promotion_last_5years', 0))
    overtime = int(employee_dict.get('overtime', 0))
    salary = str(employee_dict.get('salary_level', 'medium')).lower()
    commute = int(employee_dict.get('commute_distance_km', 15))
    
    # 1. Critical Burnout Check
    if hours > 240 and projects >= 5:
        risk_factors.append({
            "factor": "Severe Workload & Burnout",
            "severity": "CRITICAL",
            "detail": f"Working {hours} hrs/month across {projects} concurrent projects.",
            "impact": "+35% risk"
        })
    elif overtime == 1 and hours > 220:
        risk_factors.append({
            "factor": "Excessive Overtime",
            "severity": "HIGH",
            "detail": f"Frequent overtime with {hours} monthly hours.",
            "impact": "+20% risk"
        })
        
    # 2. Unrewarded High Performer (Highest Flight Risk)
    if eval_score >= 0.75 and salary == 'low' and promoted == 0:
        risk_factors.append({
            "factor": "Unrewarded High Performer",
            "severity": "CRITICAL",
            "detail": f"High performance evaluation ({eval_score:.2f}) on low compensation tier without recent promotion.",
            "impact": "+30% risk"
        })
        
    # 3. Low Job Satisfaction
    if sat < 0.35:
        risk_factors.append({
            "factor": "Acute Dissatisfaction",
            "severity": "HIGH",
            "detail": f"Reported job satisfaction level is critically low ({sat:.2f}/1.00).",
            "impact": "+28% risk"
        })
    elif sat < 0.50:
        risk_factors.append({
            "factor": "Sub-optimal Satisfaction",
            "severity": "MEDIUM",
            "detail": f"Satisfaction ({sat:.2f}) falls below enterprise benchmark (0.65).",
            "impact": "+15% risk"
        })
        
    # 4. Career Plateau / Tenure Itch
    if tenure in [3, 4, 5] and promoted == 0:
        risk_factors.append({
            "factor": "Tenure Career Plateau",
            "severity": "HIGH",
            "detail": f"{tenure} years at company without promotion; vulnerable to external recruitment.",
            "impact": "+20% risk"
        })
        
    # 5. Under-engagement / Bore-out
    if sat < 0.45 and hours < 160 and projects <= 2:
        risk_factors.append({
            "factor": "Under-engagement & Stagnation",
            "severity": "MEDIUM",
            "detail": f"Low workload ({hours} hrs, {projects} projects) paired with declining satisfaction.",
            "impact": "+18% risk"
        })
        
    # 6. Commute Strain
    if commute >= 35 and overtime == 1:
        risk_factors.append({
            "factor": "Commute & Schedule Friction",
            "severity": "MEDIUM",
            "detail": f"{commute} km commute combined with overtime creates fatigue.",
            "impact": "+12% risk"
        })
        
    return risk_factors

def generate_retention_recommendations(risk_level, risk_factors, employee_dict):
    """
    Synthesizes tailored, actionable HR management interventions based on employee's profile.
    """
    actions = []
    
    if risk_level == "LOW":
        return [
            "Maintain current engagement rhythm with quarterly 1-on-1 check-ins.",
            "Identify next-horizon skills for professional development and mentorship opportunities.",
            "Ensure recognition for ongoing project contributions."
        ]
        
    factor_names = [f["factor"] for f in risk_factors]
    
    if any("Burnout" in f or "Overtime" in f for f in factor_names):
        actions.append("Workload Audit: Immediately review project assignments; cap weekly overtime and redistribute at least one workstream.")
        actions.append("Wellness Support: Encourage mandatory compensatory time off and evaluate team resource allocation.")
        
    if any("Unrewarded" in f for f in factor_names):
        actions.append("Compensation Realignment: Fast-track a salary review to bring compensation in line with market 75th percentile.")
        actions.append("Career Advancement: Schedule an executive sponsorship meeting to outline a direct promotion roadmap within 6 months.")
        
    if any("Tenure" in f for f in factor_names):
        actions.append("Internal Mobility: Explore lateral or leadership rotation opportunities to provide fresh challenges.")
        actions.append("Retention Stay Interview: Conduct a dedicated stay interview focused on long-term career aspirations.")
        
    if any("Dissatisfaction" in f for f in factor_names):
        actions.append("Manager 1-on-1 Deep Dive: Uncover root causes of friction (management style, tooling, team dynamics).")
        actions.append("Autonomy Enhancement: Increase decision-making ownership over daily deliverables.")
        
    if any("Under-engagement" in f for f in factor_names):
        actions.append("Stretch Assignment: Re-engage with high-visibility cross-functional strategic projects.")
        
    if any("Commute" in f for f in factor_names):
        actions.append("Flexible Work Agreement: Grant 2-3 remote workdays per week to eliminate commute strain.")
        
    if not actions:
        actions.append("Conduct a proactive 30-minute stay interview to assess sentiment and career goals.")
        actions.append("Review total compensation and peer recognition opportunities.")
        
    return actions
