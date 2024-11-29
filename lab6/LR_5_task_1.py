import pandas as pd

data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

frequency_table = {}
for column in data.columns[:-1]:
    frequency_table[column] = pd.crosstab(data[column], data['Play'], normalize='all')
print("Frequency table:\n", frequency_table)

p_yes = data['Play'].value_counts(normalize=True)['Yes']
p_no = data['Play'].value_counts(normalize=True)['No']

p_overcast_yes = frequency_table['Outlook'].loc['Overcast', 'Yes']
p_high_yes = frequency_table['Humidity'].loc['High', 'Yes']
p_weak_yes = frequency_table['Wind'].loc['Strong', 'Yes']
p_yes_total = p_overcast_yes * p_high_yes * p_weak_yes * p_yes

p_overcast_no = frequency_table['Outlook'].loc['Overcast', 'No']
p_high_no = frequency_table['Humidity'].loc['High', 'No']
p_weak_no = frequency_table['Wind'].loc['Strong', 'No']
p_no_total = p_overcast_no * p_high_no * p_weak_no * p_no

p_total = p_yes_total + p_no_total
p_yes_final = p_yes_total / p_total
p_no_final = p_no_total / p_total

print(f"Probability 'Yes' for conditions: {p_yes_final:.2f}")
print(f"Probability 'No' for conditions: {p_no_final:.2f}")
