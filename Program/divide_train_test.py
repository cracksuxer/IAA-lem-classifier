import csv

with open('./F75_train.csv', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    lines = list(reader)

train_lines = lines[:2000]
test_lines = lines[2000:]


with open('train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(train_lines)

with open('test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(test_lines)
