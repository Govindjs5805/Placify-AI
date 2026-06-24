import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
num_records = 3000

# 1. Generate independent features with realistic distributions
cgpa = np.round(np.random.uniform(6.0, 10.0, num_records), 2)
projects = np.random.randint(1, 6, num_records)       # 1 to 5 projects
internships = np.random.randint(0, 4, num_records)    # 0 to 3 internships
dsa_solved = np.random.randint(0, 601, num_records)   # 0 to 600 problems
certs = np.random.randint(0, 6, num_records)          # 0 to 5 certifications
comm_score = np.random.randint(1, 11, num_records)    # 1 to 10 scale

# 2. Define a mathematical baseline for the Package (LPA)
# Base package is 3.5 LPA. Features add realistic incremental value.
base_package = 3.5

# Calculate package based on weights + a bit of random noise (variance)
# Linear Regression loves linear combinations like this:
package = (
    base_package +
    ((cgpa - 6.0) * 1.5) +          # High impact: up to +6.0 LPA for 10 CGPA
    (projects * 0.6) +              # Up to +3.0 LPA
    (internships * 1.2) +           # High impact: up to +3.6 LPA
    ((dsa_solved / 100) * 0.8) +    # High impact: up to +4.8 LPA for 600 solved
    (certs * 0.3) +                 # Up to +1.5 LPA
    ((comm_score - 1) * 0.25)       # Up to +2.25 LPA
)

# Add random real-world market noise (standard normal distribution scaled)
noise = np.random.normal(0, 0.5, num_records)
package = np.round(package + noise, 1)

# Ensure no packages drop below a realistic minimum (e.g., 3.0 LPA)
package = np.clip(package, 3.0, 45.0)

# 3. Create DataFrame
df = pd.DataFrame({
    'CGPA': cgpa,
    'Projects': projects,
    'Internships': internships,
    'DSA_Solved': dsa_solved,
    'Certifications': certs,
    'Communication_Score': comm_score,
    'Package': package
})

# 4. Save to CSV
df.to_csv('students.csv', index=False)
print(f"✅ Dataset successfully created with {num_records} rows and saved as 'students.csv'!")