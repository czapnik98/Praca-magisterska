import pandas as pd

from datapreparation import transform_input, X_test_orig
from neuralnetwork_v1 import model

baseline = {
  'month': 3,
  'city': 'warszawa',
  'type': 'blockOfFlats',
  'squareMeters': 49.0,
  'rooms': 2.0,
  'floor': 2.0,
  'floorCount': 5.0,
  'buildYear': 2000.0,
  'centreDistance': 4.0,
  'poiCount': 30.0,
  'hasParkingSpace': 'no',
  'hasBalcony': 'yes',
  'hasElevator': 'yes',
  'hasSecurity': 'no',
  'hasStorageRoom': 'no',
  'hasSchool': True,
  'hasClinic': False,
  'hasPostOffice': False,
  'hasKindergarten': True,
  'hasRestaurant': True,
  'hasCollege': False,
  'hasPharmacy': True
}

sizePlus = baseline.copy()
sizePlus['squareMeters'] = 70
sizePlus['rooms'] = 3

oldTenement = baseline.copy()
oldTenement['type'] = 'tenement'
oldTenement['buildYear'] = '1930'

cityCentre = baseline.copy()
cityCentre['centreDistance'] = 0.5

fullAmenities = baseline.copy()
fullAmenities['hasParkingSpace'] = 'yes'
fullAmenities['hasBalcony'] = 'yes'
fullAmenities['hasElevator'] = 'yes'
fullAmenities['hasSecurity'] = 'yes'
fullAmenities['hasStorageRoom'] = 'yes'

otherCity = baseline.copy()
otherCity['city'] = 'krakow'

lastMonth = baseline.copy()
lastMonth['month'] = 0

inputData = pd.DataFrame([baseline, sizePlus, oldTenement, cityCentre, fullAmenities, otherCity, lastMonth],columns=X_test_orig.columns)

inputData['price'] = model.predict(transform_input(inputData))
inputData['price'] = round(inputData['price']/100)*100
predictedData = inputData.transpose().rename({0: 'baseline', 1: 'sizePlus', 2: 'oldTenement', 3: 'cityCentre', 4: 'fullAmenities', 5: 'otherCity', 6: 'lastMonth'}, axis=1)