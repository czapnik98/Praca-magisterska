from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from datapreparation import X_train, y_train, X_test, y_test

rf1 = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 42, n_jobs=-1, max_depth=8, oob_score=True)
rf_model = rf1.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
print("Procentowy średni błąd bezwględny:", rf_mape)
rf_mae = mean_absolute_error(y_test, rf_pred)
print("Średni błąd bezwględny:", rf_mae)
rf_r2 = r2_score(y_test, rf_pred)
print("Współczynnik determinacji R2:", rf_r2)