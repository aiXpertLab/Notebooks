import pandas as pd
data = pd.read_excel("./data/default of credit card clients.xls", skiprows=1)
print(data.head())
print("rows:",data.shape[0]," columns:", data.shape[1])
data_clean = data.drop(columns=["ID","SEX"])
print(data_clean.head())
total = data_clean.isnull().sum()
percent = (data_clean.isnull().sum()/data_clean.isnull().count()*100)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

outliers = {}
for i in range(data_clean.shape[1]):
    min_t = data_clean[data_clean.columns[i]].mean() - (3 * data_clean[data_clean.columns[i]].std())
    max_t = data_clean[data_clean.columns[i]].mean() + (3 * data_clean[data_clean.columns[i]].std())
    count = 0
    for j in data_clean[data_clean.columns[i]]:
        if j < min_t or j > max_t:
            count += 1
    percentage = count/data_clean.shape[0]
    outliers[data_clean.columns[i]] = "%.3f" % percentage

print(outliers)

target = data_clean["default payment next month"]
yes = target[target == 1].count()
no = target[target == 0].count()

print("yes %: " + str(yes/len(target)*100) + " - no %: " + str(no/len(target)*100))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
plt.bar("yes", yes)
plt.bar("no", no)
ax.set_yticks([yes,no])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.show()

data_yes = data_clean[data_clean["default payment next month"] == 1]
data_no = data_clean[data_clean["default payment next month"] == 0]

over_sampling = data_yes.sample(no, replace=True, random_state=0)

data_resampled = pd.concat([data_no, over_sampling], axis=0)

target_2 = data_resampled["default payment next month"]
yes_2 = target_2[target_2 == 1].count()
no_2 = target_2[target_2 == 0].count()

print("yes %: " + str(yes_2/len(target_2)*100) + " - no %: " + str(no_2/len(target_2)*100))

data_resampled = data_resampled.reset_index(drop=True)
X = data_resampled.drop(columns=["default payment next month"])
y = data_resampled ["default payment next month"]

print(data_resampled.shape)

data_resampled = data_resampled.reset_index(drop=True)
X = data_resampled.drop(columns=["default payment next month"])
y = data_resampled["default payment next month"]

X = (X - X.min())/(X.max() - X.min())
print(X.head())

final_data = pd.concat([X, y], axis=1)
final_data.head()
final_data.to_csv("./data/a301_dccc_prepared.csv",index=False)