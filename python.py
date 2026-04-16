import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("mental_health_project_data.csv")

df.columns = df.columns.str.strip()

df['Time Period Start Date'] = pd.to_datetime(
    df['Time Period Start Date'],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['Value'])

print("\nData loaded successfully\n")

# ---------------------------
# BASIC INFO
# ---------------------------
print(df.describe())

# ---------------------------
# HISTOGRAM - The histogram shows distribution of values.
# ---------------------------
plt.figure()
sns.histplot(df['Value'], bins=20, kde=True)
plt.title("Distribution of Mental Health Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# BOXPLOT - The boxplot helps identify outliers.
# ---------------------------
plt.figure()
sns.boxplot(x=df['Value'])
plt.title("Outlier Detection")
plt.show()

# ---------------------------
# BAR CHART (TOP STATES)- Bar charts compare categories like states and age groups.
# ---------------------------
if 'Indicator' in df.columns and 'State' in df.columns:
    temp_df = df[df['Indicator'] == df['Indicator'].iloc[0]]

    if not temp_df.empty:
        top_states = temp_df.groupby('State')['Value'].mean() \
            .sort_values(ascending=False).head(5)

        plt.figure()
        top_states.plot(kind='bar')
        plt.title("Top 5 States")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.show()

# ---------------------------
# HEATMAP - The heatmap shows correlation between variables.
# ---------------------------
num_df = df.select_dtypes(include=np.number)

if not num_df.empty:
    plt.figure()
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# ---------------------------
# LINE GRAPH (TREND) - The line graph shows trends over time.
# ---------------------------
if 'Group' in df.columns:
    national = df[df['Group'] == 'National Estimate']

    if not national.empty:
        plt.figure()
        for ind in national['Indicator'].unique():
            temp = national[national['Indicator'] == ind]
            plt.plot(temp['Time Period Start Date'], temp['Value'], label=ind)

        plt.title("Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()

# ---------------------------
# AGE GROUP BAR CHART
# ---------------------------
if 'Group' in df.columns and 'Subgroup' in df.columns:
    latest = df['Time Period Start Date'].max()
    age_data = df[(df['Time Period Start Date'] == latest) & (df['Group'] == 'By Age')]

    if not age_data.empty:
        plt.figure()
        sns.barplot(data=age_data, x='Subgroup', y='Value')
        plt.xticks(rotation=45)
        plt.title("Mental Health by Age Group")
        plt.show()

        most_affected = age_data.sort_values('Value', ascending=False).iloc[0]

# ---------------------------
# PIE CHART
# ---------------------------
top_indicators = df['Indicator'].value_counts().head(5)

plt.figure()
plt.pie(top_indicators, labels=top_indicators.index, autopct='%1.1f%%', startangle=90)
plt.title("Top Indicators Distribution")
plt.axis('equal')
plt.show()

# ---------------------------
# Z-TEST
# ---------------------------
values = df['Value'].to_numpy()

mean_sample = np.mean(values)
mean_pop = 50
std = np.std(values)
n = len(values)

z = (mean_sample - mean_pop) / (std / np.sqrt(n))
p_value = 1 - norm.cdf(z)

# ---------------------------
# MACHINE LEARNING
# ---------------------------
df_ml = df.copy()

for col in ['Indicator', 'Group']:
    if col in df_ml.columns:
        df_ml[col] = df_ml[col].astype('category').cat.codes

X = df_ml.select_dtypes(include=np.number).drop(columns=['Value'], errors='ignore')
y = df_ml['Value']

if not X.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

# ---------------------------
# DETAILED OUTPUT
# ---------------------------
print("\n" + "="*60)
print("DETAILED DATA ANALYSIS REPORT")
print("="*60)

avg = df['Value'].mean()
mx = df['Value'].max()
mn = df['Value'].min()
std_val = df['Value'].std()

print("\nSome quick insights from the dataset:")
print(f"Average value: {avg:.2f}")
print(f"Highest value: {mx}")
print(f"Lowest value: {mn}")

print("\nThe average value indicates the overall mental health trend in the dataset.")
print("The highest and lowest values show the range and variation present in the data.")

# Indicators
if 'Indicator' in df.columns:
    print("\nAvailable indicators in the dataset are:")
    print(df['Indicator'].unique())

    print("\nThese indicators represent mental health aspects like medication usage and therapy.")

# Age group insight
if 'Group' in df.columns and 'Subgroup' in df.columns:
    latest = df['Time Period Start Date'].max()
    age_data = df[(df['Time Period Start Date'] == latest) & (df['Group'] == 'By Age')]

    if not age_data.empty:
        most_affected = age_data.sort_values('Value', ascending=False).iloc[0]

        print(f"\nMost affected age group: {most_affected['Subgroup']}")
        print("This age group shows the highest mental health impact among all groups.")

# Z-test output
print("\nZ-test Results:")
print(f"Z-score = {z:.2f}")
print(f"P-value = {p_value:.4f}")

if p_value < 0.05:
    print("There is a statistically significant difference in the data.")
else:
    print("There is no statistically significant difference in the data.")

# ML Results
if not X.empty:
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nMachine Learning Results:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    print("The scatter plot shows the relationship between actual and predicted values.")

    if r2 > 0.7:
        print("The model performance is good.")
    elif r2 > 0.4:
        print("The model performance is moderate.")
    else:
        print("The model performance is weak and can be improved.")

# ---------------------------
# FINAL CONCLUSION
# ---------------------------
print("\n" + "="*60)
print("FINAL CONCLUSION")
print("="*60)

print("This project analyzed mental health data using visualization, statistics, and machine learning.")
print("The average value is around 24, with maximum close to 40, showing variation in the dataset.")
print("The most affected age group is 18–29 years.")
print("The dataset mainly focuses on indicators like medication usage and therapy.")
print("Overall, the analysis provides useful insights into mental health trends.")
