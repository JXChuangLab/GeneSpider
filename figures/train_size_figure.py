import seaborn as sns

data = sns.load_dataset("tips")
print(data.head())

sns.boxplot(x=data["day"],y=data['total_bill'],hue=data["sex"])
