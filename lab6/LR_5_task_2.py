import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv')

data['insert_date'] = pd.to_datetime(data['insert_date'])

data['day_of_week'] = data['insert_date'].dt.day_name()

data = data.dropna(subset=['price'])

data['fare_category'] = pd.qcut(data['price'], q=3, labels=['low', 'medium', 'high'])

frequency_table = pd.crosstab(index=[data['train_type'], data['day_of_week']],
                              columns=data['fare_category'],
                              normalize='index')

print("\nFrequency table for fare_category & train_type & day_of_week:")
print(frequency_table)

train_type_condition = 'AVE'
day_of_week_condition = 'Friday'

try:
    p_low = frequency_table.loc[(train_type_condition, day_of_week_condition), 'low']
    p_medium = frequency_table.loc[(train_type_condition, day_of_week_condition), 'medium']
    p_high = frequency_table.loc[(train_type_condition, day_of_week_condition), 'high']

    print(f"\nProbability of lower price: {p_low:.2f}")
    print(f"Probability of medium price: {p_medium:.2f}")
    print(f"Probability of high price: {p_high:.2f}")

    if p_low > p_medium and p_low > p_high:
        prediction = 'low'
    elif p_medium > p_low and p_medium > p_high:
        prediction = 'medium'
    else:
        prediction = 'high'

    print(f"\nCategory prediction for current conditions ({train_type_condition}, {day_of_week_condition}): {prediction}")
except KeyError:
    print(f"\nImpossible to count probabilities for conditions: {train_type_condition} у {day_of_week_condition} - not enough data.")