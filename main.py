import torch
import torch.nn as nn
import io
import requests
import datetime as dt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab

#Hyperparamètre windows , epoch , hidden_layer_size


# 1) Récupération du dernier jeu de données

# Url stable du jeu de données
url_fichier = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"

# Obtention du contenu du fichier csv
fichier = requests.get(url_fichier, allow_redirects=True).content
fichier = io.StringIO(fichier.decode('utf-8'))

# Création d'un dataframe à partir du contenu du fichier
df = pd.read_csv(fichier, delimiter=";")

# (Facultatif) Sauvegarde du dataframe dans un fichier csv
df.to_csv("data-covid19-" + dt.datetime.now().strftime('%m%d%y') + ".csv")

# Aperçu du dataframe
print(df)


dataset = df
date = dataset['jour']
dataset.drop('jour',axis=1 , inplace=True)
dataset.columns = ['dep', 'sexe', 'hosp', 'rea', 'rad', 'dc']
dataset.index.name = 'date'
dataset.index = date
print(dataset)

moyenne = dataset.groupby('jour').mean()
moyenne.drop('sexe',axis=1, inplace=True)
print(moyenne)

plt.subplot(221)
plt.ylabel("Décès")
plt.xlabel("Hospitalisation")
plt.plot(moyenne['hosp'],moyenne['dc'], color="red")

plt.subplot(222)
plt.ylabel("Décès")
plt.xlabel("Réanimation")
plt.plot(moyenne['rea'],moyenne['dc'], color="blue")

plt.subplot(223)
plt.ylabel("Reanimation")
plt.xlabel("Hospitalisation")
plt.plot(moyenne['hosp'],moyenne['rea'], color="pink")
#plt.show()

plt.title("Décès en fonction du temps")
plt.xlabel("Jours")
plt.ylabel("Nombre Décès en France")
plt.xticks(rotation='vertical')
plt.plot(moyenne['dc'], color="red")
#plt.show()

from sklearn.preprocessing import MinMaxScaler

indice = np.random.permutation(moyenne.shape[0])
all_data = moyenne[['hosp', 'rea', 'dc']].values.astype(float)
print(all_data)
training_idx, test_idx = indice[:int(indice.shape[0]*0.8)], indice[int(indice.shape[0]*0.8):]

train_data = all_data[training_idx, :]
train_test = all_data[test_idx, :]

print("Affichage du vecteur test de taille :"+str(train_test.shape))
print(train_test)


scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 3))
print(train_data_normalized.shape)
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1, 3)

print("TRain data size")
print(train_data_normalized.shape)


train_window = 14


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=250, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = ( torch.zeros(1, 1, self.hidden_layer_size),
                torch.zeros(1, 1, self.hidden_layer_size))


    def forward(self, input_seq):
        print("Input_seq = ",input_seq.shape)
        print(input_seq.view(len(input_seq), 1, -1).size())
        lstm_out, self.hidden_cell = self.lstm(input_seq.view((len(input_seq), 1, 3)), self.hidden_cell)
        prediction = self.linear(lstm_out.view(len(input_seq), 1, 3))
        return prediction[-1]



model = LSTM()
loss_fonction = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_fonction(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        if i%25 == 1 :
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


fut_pred = 1
test_inputs = train_data_normalized[-train_window:, :].tolist()
model.eval()


for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())


actual_prediction = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

x = np.arange(132, 144, 1)

plt.title("Décès en fonction du temps")
plt.xlabel("Jours")
plt.ylabel("Nombre Décès en France")
plt.xticks(rotation='vertical')
plt.plot(moyenne['dc'], color="red")
plt.plot(x, actual_prediction, color="pink")
plt.show()
