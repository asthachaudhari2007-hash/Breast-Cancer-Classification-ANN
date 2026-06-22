import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="Target")
print(X.head())
print(X.shape)
print(y.shape)
print(X.info())
print(X.describe())

import pandas as pd
import plotly.express as px

target_df=pd.DataFrame({

    "Class":["Malignant","Benign"],
    "Count":[sum(y==0),sum(y==1)]
})
fig=px.bar(
    target_df,
    x="Class",
    y="Count",
    color="Class",
    title="Breast Cancer Dataset Target Distribution"
)
fig.show()
print(
    X.isnull().sum()
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(15,10))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title(" Figure Correlation ")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("Target Classes")
for value ,name in enumerate(cancer.target_names):
    print(value,"->",name)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(30,)))

model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.summary()
#train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,verbose=1)
import plotly.graph_objects as py
import pandas as pd
fig = py.Figure()
fig.add_trace(py.Scatter( y=history.history['accuracy'],  name='Training Accuracy'))
fig.update_layout(title='Training Accuracy over Epochs', xaxis_title='Epochs', yaxis_title='Accuracy')
fig.show()

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

#make prediction 
y_pred = model.predict(X_test)

# Convert probabilities into 0 and 1
y_pred_classes = (y_pred > 0.5).astype(int)

print(y_pred_classes[:10])

#Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Malignant','Benign'],
            yticklabels=['Malignant','Benign'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#Classification Report 
from sklearn.metrics import classification_report

print(classification_report(y_test,
                            y_pred_classes,
                            target_names=cancer.target_names))

#Plot training and validation accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# plot loss curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

#save the model 
model.save("breast_cancer_ann_model.h5")

#load model
from tensorflow.keras.models import load_model

loaded_model = load_model("breast_cancer_ann_model.h5")


# test on a patient data
sample = X_test[0].reshape(1,30)

prediction = loaded_model.predict(sample)

if prediction > 0.5:
    print("Benign")
else:
    print("Malignant")
