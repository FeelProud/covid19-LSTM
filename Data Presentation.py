#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import requests
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns


# In[2]:


sns.set()

# 1) Récupération du dernier jeu de données

# Url stable du jeu de données
url_fichier = "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7"

# Obtention du contenu du fichier csv
fichier = requests.get(url_fichier, allow_redirects=True).content
fichier = io.StringIO(fichier.decode('utf-8'))

# Création d'un dataframe à partir du contenu du fichier
df = pd.read_csv(fichier,delimiter=";")

# (Facultatif) Sauvegarde du dataframe dans un fichier csv
df.to_csv("data-covid19-" + dt.datetime.now().strftime('%m%d%y') + ".csv")

# Aperçu du dataframe
print(df)


# In[3]:


donnees = df[['sexe', 'hosp' , 'rea']]

Homme = donnees[(donnees['sexe'] == 1)]
Femme = donnees[(donnees['sexe'] == 2) ]

N = 2
homme_moy = (Homme.sum()["hosp"], Homme.sum()["rea"])
femme_moy = (Femme.sum()["hosp"], Femme.sum()["rea"])

ind = np.arange(N) 
width = 0.38
plt.bar(ind, homme_moy, 0.36, label='Homme', color='b', align='center')
plt.bar(ind + width, femme_moy, 0.36, label='Femme', color='r', align='center')

plt.title('Comparaison du nombre d\'hospitalisation et de réanimation par sexe')

plt.xticks(ind + width / 2, ('Hospitalisation', 'Réanimation'))
plt.legend(loc='best')
plt.show()


# In[4]:


jours = df["jour"].str.split('-')

annee = [ annee[2]  for annee in jours ]
month = [ month[1] for month in jours ]
day =   [ day[0]    for day   in jours ]

jours = []
for i in range(len(annee)):
    jours.append([annee[i],month[i],day[i]])

jours = pd.DataFrame(jours, columns=['jour', 'mois', 'année'])
donnees2 = df[["hosp", "rea", "rad", "dc"]]
tempdf = pd.concat([jours, donnees2], axis=1)
donneeParJours = tempdf.groupby(["mois","jour"]).mean()

grap = donneeParJours.reset_index()

plt.title("Evolution des cas de Covid-19 dans le temps")
plt.plot(grap["jour"],grap["hosp"],"r--" , label="Hospitalisation")
plt.plot(grap["jour"],grap["rea"],"b--" , label="Réanimation")
plt.plot(grap["jour"],grap["dc"],"y--", label="Décès")

plt.legend()
plt.show()


# In[5]:


most = df.loc[(df['sexe'] == 0)]
most = most.groupby(['dep'], as_index=False).mean()
most = most.sort_values('hosp', ascending=False).head(5)
most = most['dep'].to_list()

colour = ['b', 'g', 'r', 'c', 'm']


fig = plt.figure(1, figsize=(25, 15))

ax1 = plt.subplot(232)
ax1.set_title("Evolution des personnes en réanimation dans les 5 départements les plus touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == most[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax1.plot(temp['jour'], temp['rea'], colour[i] + "--", label=most[i])

ax2 = plt.subplot(233)
ax2.set_title("Evolution des décès dans les 5 départements les plus touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == most[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax2.plot(temp['jour'], temp['dc'], colour[i] + "--", label=most[i])

ax3 = plt.subplot(231)
ax3.set_title("Evolution des hospitalisations dans les 5 départements les plus touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == most[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax3.plot(temp['jour'], temp['hosp'], colour[i] + "--", label=most[i])


plt.setp(ax1.xaxis.get_majorticklabels(), rotation='vertical')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation='vertical')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation='vertical')

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()


# In[6]:


less = df.loc[(df['sexe'] == 0)]
less = less.groupby(['dep'], as_index=False).mean()
less = less.sort_values('hosp', ascending=True).head(5)
less = less['dep'].to_list()

colour = ['b', 'g', 'r', 'c', 'm']


fig = plt.figure(1, figsize=(25, 15))

ax1 = plt.subplot(232)
ax1.set_title("Evolution des personnes en réanimation dans les 5 départements les moins touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == less[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax1.plot(temp['jour'], temp['rea'], colour[i] + "--", label=less[i])

ax2 = plt.subplot(233)
ax2.set_title("Evolution des décès dans les 5 départements les moins touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == less[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax2.plot(temp['jour'], temp['dc'], colour[i] + "--", label=less[i])

ax3 = plt.subplot(231)
ax3.set_title("Evolution des hospitalisations dans les 5 départements les moins touchés")

for i in range (0,5):
    temp = df.loc[(df['dep'] == less[i]) & (df['sexe'] == 0)]
    temp = temp.sort_values('jour', ascending=True)
    ax3.plot(temp['jour'], temp['hosp'], colour[i] + "--", label=less[i])


plt.setp(ax1.xaxis.get_majorticklabels(), rotation='vertical')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation='vertical')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation='vertical')

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()


# In[7]:


temp = df.loc[(df['sexe'] == 0)].drop(columns=['sexe'])
temp = temp.groupby(['dep'], as_index=False).mean()

most = temp.sort_values('hosp', ascending=False).head(5)

most.reset_index(drop=True, inplace=True)

coulor = ['b', 'g', 'r', 'c', 'm']
BarName = [str(most["dep"][0]), str(most["dep"][1]), str(most["dep"][2]), str(most["dep"][3]), str(most["dep"][4])]

height = [most["hosp"][0], most["hosp"][1], most["hosp"][2], most["hosp"][3], most["hosp"][4]]
width = 0.8
plt.bar([1, 2, 3, 4, 5], height, width, color=coulor)
maximum = most["hosp"][0].max()+200

plt.xlim(0,6)
plt.ylim(0,maximum)

plt.title("Les 5 départements les plus touchés")
plt.ylabel("Moyenne du nombre d'hospitalisations")
plt.xlabel("Département")
plt.grid(True)
pylab.xticks([1, 2, 3, 4, 5], BarName)
plt.show()


# In[8]:


temp = df.loc[(df['sexe'] == 0)].drop(columns=['sexe'])
temp = temp.groupby(['dep'], as_index=False).mean()

less = temp.sort_values('hosp', ascending=True).head(5)

less.reset_index(drop=True, inplace=True)

coulor = ['b', 'g', 'r', 'c', 'm']
BarName = [str(less["dep"][0]), str(less["dep"][1]), str(less["dep"][2]), str(less["dep"][3]), str(less["dep"][4])]

height = [less["hosp"][0], less["hosp"][1], less["hosp"][2], less["hosp"][3], less["hosp"][4]]
width = 0.7
plt.bar([1, 2, 3, 4, 5], height, width, color=coulor)
maximum = less["hosp"][0].max()+10

plt.xlim(0,6)
plt.ylim(0,maximum)

plt.title("Les 5 départements les moins touchés")
plt.ylabel("Moyenne du nombre d'hospitalisations")
plt.xlabel("Département")
plt.grid(True)
pylab.xticks([1, 2, 3, 4, 5], BarName)
plt.show()


# In[9]:


nouvelle_aquitaine = ['16','17','19','23','24','33','40','47','64','79','86','87']
pays_de_la_loire = ['44','49','53','72','85']
bretagne = ['22','29','35','56']
occitanie = ['09','11','12','30','31','32','34','46','48','65','66','81','82']
normandie = ['14','27','50','61','76']
centre_val_de_loire = ['18','28','36','37','41','45']
ile_de_france = ['75','77','78','91','92','93','94','95']
hauts_de_france = ['02','59','60','62','80']
grand_est = ['08','10','51','52','54','55','57','67','68','88']
bourgogne_franche_comté = ['21','25','39','58','70','71','89','90']
auvergne_rhone_alpes = ['01','03','07','15','26','38','42','43','63','69','73','74']
PACA = ['04','05','06','13','83','84']
Corse = ['2A','2B']
Guadeloupe = ['971']
Martinique = ['972']
Guyane = ['973']
La_Réunion = ['974']
Mayotte = ['976']


# In[10]:


date = df['jour']
df.drop('jour',axis=1 , inplace=True)
df.columns = ['dep', 'sexe', 'hosp', 'rea', 'rad', 'dc']
df.index.name = 'date'
df.index = date
print(df)

moyenne = df.groupby('jour').mean()
moyenne.drop('sexe',axis=1, inplace=True)
print(moyenne)

plt.title("Hospitalisation par jours")
plt.subplot(221)
plt.ylabel("Décès")
plt.xlabel("Hospitalisation")
plt.plot(moyenne['hosp'],moyenne['dc'], color="red")
plt.subplot(222)
plt.ylabel("Décès")
plt.xlabel("Réanimation")
plt.plot(moyenne['rea'],moyenne['dc'], color="red")
plt.subplot(223)
plt.ylabel("Reanimation")
plt.xlabel("Hospitalisation")
plt.plot(moyenne['hosp'],moyenne['rea'], color="red")
plt.show()


# In[12]:


choix_dep = '17'

temp = df.loc[(df['dep'] == choix_dep) & (df['sexe'] == 0)]
print(temp)
temp = temp.sort_values('jour', ascending=True)

plt.title("Evolution des hospitalisations dans le département " + choix_dep)
plt.plot(temp.index, temp['hosp'], "r")

plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




