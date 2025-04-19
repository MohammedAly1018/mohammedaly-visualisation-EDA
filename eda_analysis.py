#libraries used
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel(r'C:\vs code\qatar-monthly-statistics-visitor-arrivals-by-mode-of-enteryecxel.xlsx')

# Rename columns to remove extra spaces and make them easier to use
df.rename(columns={
    ' Air': 'Air Arrivals',
    ' Land': 'Land Arrivals',
    ' Sea': 'Sea Arrivals',
    '  Total Visitor Arrivals': 'Total Visitor Arrivals'
}, inplace=True)

# Convert 'Month' column to datetime format for proper time series analysis
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

#Graph 1 – Line chart for visitor arrivals over time

plt.figure(figsize=(12, 6))  # Set figure size

# Plot each mode of entry over time
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

# Graph 2 – Monthly average visitor arrivals (Jan to Dec)

# Extract month number from date (1–12)
df['Month_Num'] = df['Month'].dt.month

# Grouping the data by month number and calculating average for each mode
monthly_avg = df.groupby('Month_Num')[['Air Arrivals', 'Land Arrivals', 'Sea Arrivals', 'Total Visitor Arrivals']].mean()

plt.figure(figsize=(10, 6))

# Plot monthly averages
plt.plot(monthly_avg.index, monthly_avg['Air Arrivals'], label='Air Arrivals', color='blue', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Land Arrivals'], label='Land Arrivals', color='green', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Sea Arrivals'], label='Sea Arrivals', color='orange', marker='o')
plt.plot(monthly_avg.index, monthly_avg['Total Visitor Arrivals'], label='Total Arrivals', color='red', marker='o')
plt.title('Average Monthly Visitor Arrivals by Mode of Entry and Total')
plt.xlabel('Month')
plt.ylabel('Average Number of Visitors')

# x-axis labels from Jan to Dec
plt.xticks(monthly_avg.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.tight_layout()
plt.show()

# Graph 3– Yearly growth rate of total visitor arrivals percentage

# Extract year from Month column
df['Year'] = df['Month'].dt.year

# Group total visitor arrivals by year
yearly_data = df.groupby('Year')['Total Visitor Arrivals'].sum()

# Calculating year by year % change
yearly_growth_rate = yearly_data.pct_change() * 100

plt.figure(figsize=(10, 6))
plt.plot(yearly_growth_rate.index, yearly_growth_rate, marker='o', color='blue')
plt.title('Yearly Growth Rate of Total Visitor Arrivals')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.grid(True) 
plt.tight_layout()
plt.show()
