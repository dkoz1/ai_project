import csv

db = {}
i = 0
labelki = {}
with open('przetworzone_dane.csv', 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=';')
    for row in r:
        if i == 0:
            labelki[row[0]] = row[1]
            i += 1
            continue
        db[row[0]] = row[1]
year = 0
month = 0
day = 0
for key in db.keys():
    year = key.split('-')[0]
    month = key.split('-')[1]
    day = key.split('-')[2]
    break

with open('przetworzone_dane2.csv', 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=';')
    # 1984-12-21 - poczatek
    # 2018-11-07 - koniec
    w.writerow(['Data_zgloszenia', 'Ilosc_zgloszen'])
    for year_ in range(1984, 2019):
        for month_ in range(1, 13):
            if year_ == 1984 and month_ != 12:
                continue
            for day_ in range(1, 32):
                if month_ in [4, 5, 6, 9, 11] and day_ == 31:
                    break
                if month_ == 2 and (year_ % 4 == 0 and year_ % 100 != 0
                                    or year_ % 400 == 0):
                    if day_ >= 30:
                        break
                elif month_ == 2 and day_ >= 29:
                    break
                data = '%04d-%02d-%02d' % (year_, month_, day_)
                if data in db.keys():
                    w.writerow([data, db[data]])
                else:
                    w.writerow([data, 0])