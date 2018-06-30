# -*- coding: utf-8 -*-

from __future__ import print_function

from time import time

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

#import pandas as pd

print("Veri yukleniyor...")

kisi = fetch_lfw_people('./faces', min_faces_per_person = 100 , resize = 1.5)

print('Yuklendi!')

n_samples, h, w = kisi.images.shape

print(h)
print(w)

#print(kisi.data)

X = kisi.data
n_features = X.shape[1]

print(n_features)

y = kisi.target
target_names = kisi.target_names
n_classes = target_names.shape[0]


print("Toplam veri boyutu: ")
print("n_samples:%d " % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

#training ve testing kumesini bolme islemi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#train kümesi veriyi alır ve en iyi çıktı alınmak üzere onu 'eğitir'.
#test kümesi train kümesindeki verinin ne kadar iyi bir şekilde 'eğitildiğini' test ederek
#çıktı verir ve tahmin işleminin iyi bir şekilde gerçekleştirilmesine yardımcı olur.
 
# yuz veritabanindaki ozyuzleri hesaplama
n_components = 150
 
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

#randomizedpca X_train verilerinin lineer boyutunu düşürür ve 
#pca depişkenine atar.amaç özyüzleri oluşturmaktır. 
 
eigenfaces = pca.components_.reshape((n_components, h, w))
X_train_pca = pca.transform(X_train)

def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4):
    plt.figure(figsize=(1.8*n_col, 2.4*n_row))
    plt.subplots_adjust(bottom = 0, left=.01, right = .99, top = .90, hspace = .35)
    
    for i in range(n_row*n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())
        
eigenface_titles = ["ozyuzler: %d" %i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

print("Siniflandiricilarin gelistirme kumesine yerlestirilmesi\n")

zaman = time()

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1] }
              
              
#param_grid bir sozlüktür ve burada sınıflandırıcıları anahtar olarak almıştır.
#GridSearchCV tahmin etmeyi gerçekleştiren bir fonksiyondur. param_grid sözlüğündeki
#sınıflandırıcıları parametre olarak almıştır.

 
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)

print("%0.3fs icinde islem gerceklestirildi!" % (time() - zaman))
print("Orgu aramasi ile en iyi tahmin edicinin bulunmasi: ")
print(clf.best_estimator_)
 

X_test_pca = pca.transform(X_test)
y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))

#print('Confusion Matrix')


#cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
#df = pd.DataFrame(cm, columns = target_names, index = target_names)
#print(df)

print(y_pred)
print(y_pred.shape[0])


#son olarak tahmin edilen isimler ve gercek isimler karsilastirilir

def isim(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'tahmin edilen: %s\ngercek:     %s'%(pred_name, true_name)
 
tahmin = [isim(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
 
plot_gallery(X_test, tahmin, h, w, 6, 4)




