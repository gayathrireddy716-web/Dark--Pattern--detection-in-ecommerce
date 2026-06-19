# DARK PATTERN ANALYSIS - FINAL

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Excel file
df = pd.read_excel('dark_pattern_dataset_1000.xlsx') #initial error due to the file location of collab but running on powerhsell so file location error !!!

print("✅ Dataset Loaded\n")
print(df.head())

# 2. Cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.fillna(0)

print("\n✅ Data Cleaned\n")

# 3. Analysis
print("\n📊 Pattern Count:\n", df['pattern_label'].value_counts())


# 4. Graph
df['pattern_label'].value_counts().plot(kind='bar')
plt.title("Dark vs Normal Patterns")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# 5. Machine Learning
X = df[['urgency_words','popup_count','price_difference','button_highlight']]
y = df['pattern_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#channges in the RFC model as we are facing class imbalance issue in the dataset so to solve that we will use class_weight='balanced' in the RandomForestClassifier which will automatically adjust weights inversely proportional to class frequencies
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\n🤖 Model Accuracy:", accuracy)

# 6. Prediction
sample = [[2, 3, 50, 1]]
prediction = model.predict(sample)

print("\n🔮 Prediction (0=Normal, 1=Dark):", prediction) # change in the prediction as we took platform column earlier which wasnt present in the dataset earlier so solved that

# 7. Save cleaned file
df.to_excel('cleaned_data.xlsx', index=False) #cleaned process data clearly and save it in a new file called cleaned_data.xlsx

print("\n Cleaned file saved!")
print("\n DONE SUCCESSFULLY !!!!")

output:
✅ Dataset Loaded

   id                     page_text  urgency_words  popup_count  \
0   1     No urgency normal listing              0            0   
1   2    Flash sale ends soon hurry              4            4   
2   3  Only 2 left grab immediately              3            3   
3   4   Limited time offer act fast              4            2   
4   5  Only 2 left grab immediately              3            0   

   price_difference  button_highlight pattern_label  
0                19                 0        Normal  
1                45                 0          Dark  
2                82                 0          Dark  
3                17                 0          Dark  
4                 6                 0          Dark  

✅ Data Cleaned


📊 Pattern Count:
 pattern_label
Dark      924
Normal     76
Name: count, dtype: int64



🤖 Model Accuracy: 1.0

🔮 Prediction (0=Normal, 1=Dark): ['Dark']

 Cleaned file saved!

 DONE SUCCESSFULLY !!!!
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
  warnings.warn(
