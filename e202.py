import pandas as pd

data = pd.read_csv('./data/energydata_complete.csv')
data = data.drop(columns = ['date'])
print(data.head())

cols = data._get_numeric_data().columns
data.isnull().sum()

outliers = {}
for i in range(data.shape[1]):
    min_t = data[data.columns[i]].mean() \
            - (3 * data[data.columns[i]].std())
max_t = data[data.columns[i]].mean() \
        + (3 * data[data.columns[i]].std())
count = 0
for j in data[data.columns[i]]:
    if j < min_t or j > max_t:
        count += 1
percentage = count / data.shape[0]
outliers[data.columns[i]] = "%.3f" % percentage
print(outliers)