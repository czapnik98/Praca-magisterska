from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from datapreparation import X_train, y_train, X_test, y_test

lr = linear_model.LinearRegression(fit_intercept = True)
lr_model = lr.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mape = mean_absolute_percentage_error(y_test, lr_pred)
print("Procentowy średni błąd bezwględny:", lr_mape)
lr_mae = mean_absolute_error(y_test, lr_pred)
print("Średni błąd bezwględny:", lr_mae)
lr_r2 = r2_score(y_test, lr_pred)
print("Współczynnik determinacji R2:", lr_r2)
