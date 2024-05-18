import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('ergonomic_risk_scores.csv')

# Set the style for seaborn
sns.set(style="whitegrid")

# Create a histogram for the 'Ergonomic Risk Score'
plt.figure(figsize=(10, 6))
sns.histplot(df['Ergonomic Risk Score'], kde=False, bins=10, color='skyblue')
plt.title('Distribution of Ergonomic Risk Scores')
plt.xlabel('Ergonomic Risk Score')
plt.ylabel('Frequency')
plt.savefig('ergonomic_risk_score_distribution.png')
plt.close()

# Create a count plot for the 'Action Level'
plt.figure(figsize=(10, 6))
sns.countplot(y='Action Level', data=df, palette='viridis')
plt.title('Count of Action Levels')
plt.xlabel('Count')
plt.ylabel('Action Level')
plt.savefig('action_level_count.png')
plt.close()
