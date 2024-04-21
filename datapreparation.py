import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder

df1 = pd.read_csv('apartments_pl_2024_01.csv')
df2 = pd.read_csv('apartments_pl_2024_02.csv')
df3 = pd.read_csv('apartments_pl_2024_03.csv')
df1['month'] = 0
df2['month'] = 1
df3['month'] = 2
df = pd.concat([df1, df2, df3])
df.drop(['condition', 'buildingMaterial'], axis=1, inplace=True)

cleanData = df.copy()
cleanData = cleanData.drop(['ownership', 'longitude', 'latitude'], axis=1)
cleanData = cleanData.dropna()

np.random.seed(3981)
guessData = df.copy()
# zgadywanie brakujących danych
# kamienice wybudowane między 1875 a 1975
guessData['buildYear'] = guessData.apply(lambda x: round(np.random.normal(loc=1925, scale=25))
if (pd.isna(x['buildYear']) and x['type'] == 'tenement') else x['buildYear'], axis=1)
# bloki wybudowane między 1975 a 2000
guessData['buildYear'] = guessData.apply(lambda x: round(np.random.normal(loc=1991, scale=15))
if (pd.isna(x['buildYear']) and x['type'] == 'blockOfFlats') else x['buildYear'], axis=1)
# apartamentowce wybudowane po 2000
guessData['buildYear'] = guessData.apply(lambda x: (2024 - round(abs(np.random.normal(loc=0, scale=15))))
if (pd.isna(x['buildYear']) and x['type'] == 'apartmentBuilding') else x['buildYear'], axis=1)
# zgadywanie typu: kamienica, wybudowane przed 1960
guessData['type'] = guessData.apply(lambda x: 'tenement' if (pd.isna(x['type']) and x['buildYear'] < 1960)
else x['type'], axis=1)
# zgadywanie typu: blok, wybudowane między 1960 a 2000
guessData['type'] = guessData.apply(lambda x: 'blockOfFlats' if (pd.isna(x['type']) and x['buildYear'] < 2000)
else x['type'], axis=1)
# zgadywanie typu: blok, należy do spółdzielni
guessData['type'] = guessData.apply(lambda x: 'blockOfFlats' if (pd.isna(x['type']) and x['ownership'] == 'cooperative')
else x['type'], axis=1)
# zgadywanie typu: blok lub apartamentowiec, wybudowany po 2000
guessData['type'] = guessData.apply(lambda x: ('blockOfFlats' if(np.random.randint(0, 2) == 0)
                                               else 'apartmentBuilding') if (pd.isna(x['type'])) else x['type'], axis=1)
# najwięcej jest budynków z 4 piętrami
guessData['floorCount'] = guessData.apply(lambda x: max(x['floor'], 4) if pd.isna(x['floorCount']) else x['floorCount'], axis=1)
guessData['floorCount'] = guessData.apply(lambda x: 4 if pd.isna(x['floorCount']) else x['floorCount'], axis=1)
# równomierne rozmieszczenia apartamentów na wszystkich piętrach
guessData['floor'] = guessData.apply(lambda x: np.random.randint(1, x['floorCount']+1)
if pd.isna(x['floor']) else x['floor'], axis=1)
# windy montowane w budynkach z piętrami większymi od 4
guessData['hasElevator'] = guessData.apply(lambda x: 'no' if (pd.isna(x['hasElevator']) and x['floorCount'] <= 4)
else x['hasElevator'], axis=1)
guessData['hasElevator'] = guessData.apply(lambda x: 'yes' if (pd.isna(x['hasElevator']) and x['floorCount'] > 4)
else x['hasElevator'], axis=1)

guessData = guessData.dropna()
guessData = guessData.drop(['ownership', 'longitude', 'latitude'], axis=1)
clean_df = guessData.drop_duplicates().reset_index(drop=True)
clean_df.drop(['id'], axis=1, inplace=True)

cut_df = clean_df[clean_df['squareMeters'] <= 110]
Q1 = np.percentile(cut_df['price'], 25)
Q3 = np.percentile(cut_df['price'], 75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR
cut_df = cut_df[(cut_df['price'] <= upper_bound)]

pois = ['School', 'Clinic', 'PostOffice', 'Kindergarten', 'Restaurant', 'College', 'Pharmacy']
poiDistanceThreshold = 0.5
for poi in pois:
    cut_df['has' + poi] = cut_df[poi[0].lower() + poi[1:] + "Distance"].transform(lambda x: True if x <= poiDistanceThreshold else False)
    cut_df = cut_df.drop([poi[0].lower() + poi[1:] + "Distance"], axis=1)

cut_df['price'] = cut_df.pop('price')

x_set = cut_df.loc[:, cut_df.columns != 'price']
y_set = cut_df['price'].values

X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=31413)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=32056)
ct = make_column_transformer(
  (MinMaxScaler(), ['month', 'squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 'centreDistance', 'poiCount']),
  (OneHotEncoder(), ['city', 'type']),
  (OrdinalEncoder(), ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom', 'hasSchool', 'hasClinic', 'hasPostOffice', 'hasKindergarten', 'hasRestaurant', 'hasCollege', 'hasPharmacy'])
)
ct.fit(X_train)


def transform_input(X_input):
    return ct.transform(X_input)


X_train_orig = X_train.copy()
X_valid_orig = X_valid.copy()
X_test_orig = X_test.copy()

X_train = transform_input(X_train)
X_valid = transform_input(X_valid)
X_test = transform_input(X_test)


