import pandas as pd

from datapreparation import X_test_orig, transform_input
from neuralnetwork_v1 import model


def get_user_input():
    user_input = {}

    # Atrybuty, dla których użytkownik wprowadza wartości liczbowe
    numeric_attributes = ['month', 'squareMeters', 'rooms', 'floor', 'floorCount',
                          'buildYear', 'centreDistance', 'poiCount']

    # Atrybuty, które mogą przyjmować wartości 'tak' lub 'nie'
    binary_attributes = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']

    # Atrybuty, które mogą przyjmować wartości True lub False
    boolean_attributes = ['hasSchool', 'hasClinic', 'hasPostOffice', 'hasKindergarten', 'hasRestaurant',
                          'hasCollege', 'hasPharmacy']

    # Atrybut 'city' - nazwa miasta
    city = input("Podaj nazwę miasta: ").lower()
    user_input['city'] = city

    # Atrybut 'type' - rodzaj budynku
    building_type = input("Podaj rodzaj budynku ('blok', 'apartamentowiec', 'kamienica'): ").lower()
    if building_type == 'blok':
        user_input['type'] = 'blockOfFlats'
    elif building_type == 'apartamentowiec':
        user_input['type'] = 'apartmentBuilding'
    elif building_type == 'kamienica':
        user_input['type'] = 'tenement'
    else:
        print("Nieprawidłowy rodzaj budynku. Wprowadź 'blok', 'apartamentowiec' lub 'kamienica'.")
        return None

    # Wprowadzanie danych przez użytkownika
    for attr in numeric_attributes:
        value = input(f"Podaj wartość dla atrybutu '{attr}': ")
        # Sprawdzenie czy wartość jest możliwa do przekonwertowania na liczbę
        try:
            user_input[attr] = float(value)
        except ValueError:
            print(f"Nieprawidłowa wartość dla atrybutu '{attr}'. Wprowadź liczbę.")
            return None

    # Wprowadzanie danych dla atrybutów 'tak' lub 'nie'
    for attr in binary_attributes:
        value = input(f"Czy '{attr}' jest obecne (wprowadź 'tak' lub 'nie'): ").lower()
        # Zamiana odpowiedzi użytkownika na 'yes' lub 'no'
        if value == 'tak':
            user_input[attr] = 'yes'
        elif value == 'nie':
            user_input[attr] = 'no'
        else:
            print(f"Nieprawidłowa odpowiedź. Wprowadź 'tak' lub 'nie' dla atrybutu '{attr}'.")
            return None

    # Wprowadzanie danych dla atrybutów True lub False
    for attr in boolean_attributes:
        value = input(f"Czy '{attr}' jest obecne (wprowadź 'tak' lub 'nie'): ").lower()
        # Zamiana odpowiedzi użytkownika na True lub False
        if value == 'tak':
            user_input[attr] = True
        elif value == 'nie':
            user_input[attr] = False
        else:
            print(f"Nieprawidłowa odpowiedź. Wprowadź 'tak' lub 'nie' dla atrybutu '{attr}'.")
            return None

    return user_input


# Wywołanie funkcji i pobranie danych od użytkownika
user_data = get_user_input()

# Jeśli dane nie zostały poprawnie wprowadzone, powtórz próbę
while user_data is None:
    user_data = get_user_input()

# Wyświetlenie wprowadzonych przez użytkownika danych
print("\nWprowadzone dane:")
for key, value in user_data.items():
    print(f"{key}: {value}")
input_data = pd.DataFrame([user_data],columns=X_test_orig.columns)
input_data['price'] = model.predict(transform_input(input_data))
input_data['price'] = round(input_data['price']/100)*100
print("Przewidywana cena wynosi",input_data.loc[0,'price'], "zł")