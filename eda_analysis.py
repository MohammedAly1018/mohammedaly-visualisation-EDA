import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN

"""Dataset importing"""

df = pd.read_excel(r'C:\vs code\qatar-monthly-statistics-visitor-arrivals-by-mode-of-enteryecxel.xlsx')

df

df.rename(columns={
    ' Air': 'Air Arrivals',
    ' Land': 'Land Arrivals',
    ' Sea': 'Sea Arrivals',
    '  Total Visitor Arrivals': 'Total Visitor Arrivals'
}, inplace=True)


df.head()



"""Data Quality Check (Missing Values, Duplicates)"""

missing_values = df.isnull().sum()


duplicate_rows = df.duplicated().sum()


df_info = df.info()

missing_values, duplicate_rows, df_info

"""Summary Statistics and Descriptive Analysis"""

descriptive_stats = df.describe()


df_head = df.head()

# Display the summary statistics and data preview
descriptive_stats, df_head

"""Data Visualization (Distribution of Columns)"""

# Plot histograms for each numerical column to check distributions
plt.figure(figsize=(12, 8))

# Air Arrivals distribution
plt.subplot(2, 2, 1)
plt.hist(df['Air Arrivals'], bins=30, color='blue', alpha=0.7)
plt.title('Air Arrivals Distribution')
plt.xlabel('Air Arrivals')
plt.ylabel('Frequency')


plt.tight_layout()
plt.show()

# Land Arrivals distribution
plt.subplot(2, 2, 2)
plt.hist(df['Land Arrivals'], bins=30, color='green', alpha=0.7)
plt.title('Land Arrivals Distribution')
plt.xlabel('Land Arrivals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Sea Arrivals distribution
plt.subplot(2, 2, 3)
plt.hist(df['Sea Arrivals'], bins=30, color='orange', alpha=0.7)
plt.title('Sea Arrivals Distribution')
plt.xlabel('Sea Arrivals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Total Visitor Arrivals distribution
plt.subplot(2, 2, 4)
plt.hist(df['Total Visitor Arrivals'], bins=30, color='red', alpha=0.7)
plt.title('Total Visitor Arrivals Distribution')
plt.xlabel('Total Visitor Arrivals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

"""Correlation analysis"""

correlation_matrix = df[['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']].corr()


correlation_matrix

"""Anomaly Detection (Using Z-Score and Boxplot)"""

z_scores = df[['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']].apply(zscore)

# Set a threshold for detecting outliers (z-score > 3 or z-score < -3)
outliers_zscore = (z_scores > 3) | (z_scores < -3)

# Get rows with outliers based on Z-score
outlier_rows_zscore = df[outliers_zscore.any(axis=1)]

# Display outlier rows
outlier_rows_zscore[['Month', 'Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']]

# Create boxplots for each column to visualize the distribution and potential outliers
plt.figure(figsize=(12, 8))

# Air Arrivals boxplot
plt.subplot(2, 2, 1)
sns.boxplot(x=df['Air Arrivals'])
plt.title('Air Arrivals Boxplot')

# Land Arrivals boxplot
plt.subplot(2, 2, 2)
sns.boxplot(x=df['Land Arrivals'])
plt.title('Land Arrivals Boxplot')

# Sea Arrivals boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x=df['Sea Arrivals'])
plt.title('Sea Arrivals Boxplot')

# Total Visitor Arrivals boxplot
plt.subplot(2, 2, 4)
sns.boxplot(x=df['Total Visitor Arrivals'])
plt.title('Total Visitor Arrivals Boxplot')

plt.tight_layout()
plt.show()

""" Null Value Handling and Missing Data"""

# Check if there are any null values in each column
missing_values = df.isnull().sum()

# Handle missing values (for simplicity, we will drop rows with missing data)
df_cleaned = df.dropna()

# Alternatively, you can fill missing values with the mean (or other methods) if required
# df_cleaned = df.fillna(df.mean())

# Display cleaned data
df_cleaned_info = df_cleaned.info()

missing_values, df_cleaned_info

"""Check Data Types and Convert if Necessary"""

# Check data types of all columns
data_types = df.dtypes

# If any columns need type conversion (e.g., 'Month' should be datetime)
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

# Verify if data types are correct
data_types_updated = df.dtypes

data_types, data_types_updated

df

"""Visualize the Data Over Time (Trend Analysis)"""

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Air Arrivals'], label='Air Arrivals', color='blue')
plt.plot(df['Month'], df['Land Arrivals'], label='Land Arrivals', color='green')
plt.plot(df['Month'], df['Sea Arrivals'], label='Sea Arrivals', color='orange')
plt.plot(df['Month'], df['Total Visitor Arrivals'], label='Total Arrivals', color='red')

plt.title('Visitor Arrivals by Mode of Entry (Air, Land, Sea) and Total Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Visitors')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""Perform Correlation Analysis"""

correlation_matrix = df[['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']].corr()

# Display the correlation matrix
correlation_matrix

"""Seasonal Trend Analysis"""

# Extracting month and year from 'Month' column for seasonal analysis
df['Year'] = df['Month'].dt.year
df['Month_Num'] = df['Month'].dt.month

# Calculating the average visitors for each month across all years
monthly_avg = df.groupby('Month_Num')[['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']].mean()

# Plotting seasonal trends for each type of arrival
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg.index, monthly_avg['Air Arrivals'], label='Air Arrivals', color='blue', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Land Arrivals'], label='Land Arrivals', color='green', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Sea Arrivals'], label='Sea Arrivals', color='orange', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Total Visitor Arrivals'], label='Total Arrivals', color='red', marker='o')

plt.title('Average Monthly Visitor Arrivals by Mode of Entry and Total')
plt.xlabel('Month')
plt.ylabel('Average Number of Visitors')
plt.xticks(monthly_avg.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.tight_layout()
plt.show()

"""Ttest"""

# Define the months to analyze
jan_2023 = df[df['Month'] == '2023-01-01'][['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']]
dec_2019 = df[df['Month'] == '2019-12-01'][['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']]
jan_2021 = df[df['Month'] == '2021-01-01'][['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']]

# Perform t-tests comparing these months against the rest of the data for each arrival type
normal_data = df[(df['Month'] != '2023-01-01') & (df['Month'] != '2019-12-01') & (df['Month'] != '2021-01-01')]

# Perform t-test for each entry mode and total arrivals
t_tests = {}
for col in ['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']:
    jan_2023_test = ttest_ind(jan_2023[col], normal_data[col], alternative='two-sided')
    dec_2019_test = ttest_ind(dec_2019[col], normal_data[col], alternative='two-sided')
    jan_2021_test = ttest_ind(jan_2021[col], normal_data[col], alternative='two-sided')

    t_tests[col] = {
        'Jan 2023 p-value': jan_2023_test.pvalue,
        'Dec 2019 p-value': dec_2019_test.pvalue,
        'Jan 2021 p-value': jan_2021_test.pvalue
    }

# Display p-values for the t-tests
t_tests

import matplotlib.pyplot as plt
import numpy as np

# Extract the p-values from the t-test results for the three months
labels = ['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']

# Extract p-values for each of the three months (Jan 2023, Dec 2019, Jan 2021)
jan_2023_p_values = [t_tests['Air Arrivals']['Jan 2023 p-value'],
                     t_tests['Land Arrivals']['Jan 2023 p-value'],
                     t_tests['Sea Arrivals']['Jan 2023 p-value'],
                     t_tests['Total Visitor Arrivals']['Jan 2023 p-value']]

dec_2019_p_values = [t_tests['Air Arrivals']['Dec 2019 p-value'],
                     t_tests['Land Arrivals']['Dec 2019 p-value'],
                     t_tests['Sea Arrivals']['Dec 2019 p-value'],
                     t_tests['Total Visitor Arrivals']['Dec 2019 p-value']]

jan_2021_p_values = [t_tests['Air Arrivals']['Jan 2021 p-value'],
                     t_tests['Land Arrivals']['Jan 2021 p-value'],
                     t_tests['Sea Arrivals']['Jan 2021 p-value'],
                     t_tests['Total Visitor Arrivals']['Jan 2021 p-value']]

# Create a bar chart to visualize the p-values for all three months
x = np.arange(len(labels))  # The label locations
width = 0.25  # The width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars for the p-values of 2023, 2019, and 2021
bars1 = ax.bar(x - width, jan_2023_p_values, width, label='Jan 2023')
bars2 = ax.bar(x, dec_2019_p_values, width, label='Dec 2019')
bars3 = ax.bar(x + width, jan_2021_p_values, width, label='Jan 2021')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Arrival Mode')
ax.set_ylabel('p-values')
ax.set_title('T-Test p-values for Anomalous Months (Jan 2023, Dec 2019, Jan 2021)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Extract the p-values from the t-test results for all three years
labels = ['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']

# Extract p-values for each of the three years
jan_2023_p_values = [t_tests['Air Arrivals']['Jan 2023 p-value'],
                     t_tests['Land Arrivals']['Jan 2023 p-value'],
                     t_tests['Sea Arrivals']['Jan 2023 p-value'],
                     t_tests['Total Visitor Arrivals']['Jan 2023 p-value']]

dec_2019_p_values = [t_tests['Air Arrivals']['Dec 2019 p-value'],
                     t_tests['Land Arrivals']['Dec 2019 p-value'],
                     t_tests['Sea Arrivals']['Dec 2019 p-value'],
                     t_tests['Total Visitor Arrivals']['Dec 2019 p-value']]

jan_2021_p_values = [t_tests['Air Arrivals']['Jan 2021 p-value'],
                     t_tests['Land Arrivals']['Jan 2021 p-value'],
                     t_tests['Sea Arrivals']['Jan 2021 p-value'],
                     t_tests['Total Visitor Arrivals']['Jan 2021 p-value']]

# Create a bar chart to visualize the p-values for all three anomalous months
x = np.arange(len(labels))  # The label locations
width = 0.25  # The width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars for the p-values of 2019, 2021, and 2023
bars1 = ax.bar(x - width, jan_2023_p_values, width, label='Jan 2023')
bars2 = ax.bar(x, dec_2019_p_values, width, label='Dec 2019')
bars3 = ax.bar(x + width, jan_2021_p_values, width, label='Jan 2021')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Arrival Mode')
ax.set_ylabel('p-values')
ax.set_title('Mann-Whitney U Test p-values for Anomalous Months')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

"""Mann-Whitney U Test for Anomalous Data"""

# Define the list of anomalous months
anomalous_months = ['2023-01-01', '2019-12-01', '2021-01-01']

# Initialize a dictionary to store the results
mann_whitney_results = {}

# Loop over the anomalous months
for anomaly in anomalous_months:
    # Extract the data for the anomalous month
    anomaly_data = df[df['Month'] == anomaly]

    # Extract the normal data (exclude the anomalous months)
    normal_data = df[df['Month'] != anomaly]

    # Perform the Mann-Whitney U test for each column (Air, Land, Sea, Total Arrivals)
    anomaly_results = {}

    for col in ['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']:
        # Perform the Mann-Whitney U test for the current column
        u_test_result = mannwhitneyu(anomaly_data[col], normal_data[col], alternative='two-sided')

        # Store the p-value for each test
        anomaly_results[col] = u_test_result.pvalue

    # Store the results for the current anomalous month
    mann_whitney_results[anomaly] = anomaly_results

# Display the Mann-Whitney U test results for each anomalous month
mann_whitney_results

# Prepare data for plotting
labels = ['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']
anomalous_months = ['2023-01-01', '2019-12-01', '2021-01-01']  # Anomalous months (2023, 2019, 2021)

# Prepare lists to store the p-values for each anomalous month
p_values_2023 = [mann_whitney_results['2023-01-01'][col] for col in labels]
p_values_2019 = [mann_whitney_results['2019-12-01'][col] for col in labels]
p_values_2021 = [mann_whitney_results['2021-01-01'][col] for col in labels]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width and positioning
bar_width = 0.25
index = np.arange(len(labels))

# Create bars for each anomalous month
bar1 = ax.bar(index, p_values_2023, bar_width, label='January 2023')
bar2 = ax.bar(index + bar_width, p_values_2019, bar_width, label='December 2019')
bar3 = ax.bar(index + 2 * bar_width, p_values_2021, bar_width, label='January 2021')

# Labeling the plot
ax.set_xlabel('Arrival Mode')
ax.set_ylabel('p-values')
ax.set_title('Mann-Whitney U Test p-values for Anomalous Months (2019, 2021, 2023)')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

"""Yearly Growth Rate (Percentage Change)"""

df['Year'] = df['Month'].dt.year
yearly_data = df.groupby('Year')['Total Visitor Arrivals'].sum()

# Calculate percentage change from the previous year
yearly_growth_rate = yearly_data.pct_change() * 100
yearly_growth_rate

df['Year'] = df['Month'].dt.year
yearly_data = df.groupby('Year')['Total Visitor Arrivals'].sum()

# Calculate percentage change from the previous year
yearly_growth_rate = yearly_data.pct_change() * 100

# Plot the yearly growth rate
plt.figure(figsize=(10, 6))
plt.plot(yearly_growth_rate.index, yearly_growth_rate, marker='o', color='b', label='Yearly Growth Rate')
plt.title('Yearly Growth Rate of Total Visitor Arrivals')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

df
