import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

from datapreparation import X_train, y_train, X_valid, y_valid, y_test, X_test

tf.random.set_seed(42024)
tf.keras.utils.set_random_seed(42429)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
  tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanAbsolutePercentageError(),
    metrics=['mae']
)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(X_train, y_train, epochs=500, batch_size=4, validation_data=(X_valid, y_valid), callbacks=[callback])
y_Pred = model.predict(X_test)
nn_mape = mean_absolute_percentage_error(y_test, y_Pred)
print("Procentowy średni błąd bezwględny:", nn_mape)
nn_mae = mean_absolute_error(y_test, y_Pred)
print("Średni błąd bezwględny:", nn_mae)
nn_r2 = r2_score(y_test, y_Pred)
print("Współczynnik determinacji R2:", nn_r2)