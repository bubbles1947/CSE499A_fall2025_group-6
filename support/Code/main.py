import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('trainUJL_edgeimpulse.csv')
val_df   = pd.read_csv('validationUJL_edgeimpulse.csv')

X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values

X_val = val_df.drop(columns=['label']).values
y_val = val_df['label'].values

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
num_classes = len(le.classes_)

teacher = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

teacher.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

teacher.fit(X_train, y_train_enc, validation_data=(X_val, y_val_enc),
            epochs=20, batch_size=32)

student = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

student.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#compile student for evaluation on hard labels
student.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

student_loss, student_acc = student.evaluate(X_val, y_val_enc, verbose=0)
print(f"Student accuracy before distillation: {student_acc:.4f}")

teacher_preds = teacher.predict(X_train)
student.compile(optimizer='adam',
                loss=tf.keras.losses.KLDivergence())
student.fit(X_train, teacher_preds, epochs=10, batch_size=32, verbose=1)

#re-compile student with hard-label loss and metrics
student.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

student_loss, student_acc = student.evaluate(X_val, y_val_enc, verbose=0)
print(f"Student accuracy after distillation: {student_acc:.4f}")

teacher.save('teacher_model.h5')
student.save('student_model.h5')
print("teacher and student models saved.")

converter = tf.lite.TFLiteConverter.from_keras_model(student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('studentModelQuant.tflite', 'wb') as f:
    f.write(tflite_model)

print("quantized TFLite student model saved as 'studentModelQuant.tflite'")
