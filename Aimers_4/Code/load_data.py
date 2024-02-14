import pandas as pd
import numpy as np

submission = pd.read_csv('C:/Users/user/PycharmProjects/pythonProject3/Aimers_4/Dataset/submission.csv')
print("[submission]")
print(submission)

train = pd.read_csv('C:/Users/user/PycharmProjects/pythonProject3/Aimers_4/Dataset/train.csv')
print("\n[train]")
print(train.head(5))
print(train.columns)

save = pd.read_csv('C:/Users/user/PycharmProjects/pythonProject3/Aimers_4/Dataset/save.csv')
print("\n[save]")
print(save)

save2 = pd.read_csv('C:/Users/user/PycharmProjects/pythonProject3/Aimers_4/Dataset/save2.csv')
print("\n[save]")
print(save2)