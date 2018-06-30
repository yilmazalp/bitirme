#from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
#from pylab import plot,subplot,axis,stem,show,figure


from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort
from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import matplotlib.pyplot as plt



def princomp(A, numpc = 0):
    #kovaryans matriksin ozdegerlerini ve ozvektorlerini hesapla
    M = (A-mean(A.T, axis = 1)).T #ortalamayı çıkartıp M değişkenine ata
    [latent,coeff] = linalg.eig(cov(M))
    
    p = size(coeff,axis=1)
    idx = argsort(latent) # özdeğerleri küçükten büyüğe sırala
    idx = idx[::-1]       
    
    coeff = coeff[:,idx]
    latent = latent[idx] # özdeğerleri sırala
    
    if numpc < p and numpc >= 0:
        coeff = coeff[:,range(numpc)] # ihtiyaç duyulduğunda bazı temel bileşenleri sil

    score = dot(coeff.T,M) # yeni uzaydaki verinin izdüşümü
    
    return coeff,score,latent
    
A = imread('turing.jpg') # imaj doyasını yükle
A = mean(A,2) # iki boyutlu diziye ata

full_pc = size(A,axis=1) # bütün temel bileşenlerin sayısı
i = 1
dist = []

for numpc in range(0,full_pc+10,10): # 0 10 20 ... full_pc + 10 sayısı içinde döngü
    coeff, score, latent = princomp(A,numpc)
    
    Ar = dot(coeff,score).T + mean(A,axis=0) # imajı tekrar yapılandırma
    
    # Frobenius normdaki farklılık
    dist.append(linalg.norm(A-Ar,'fro'))
    
    # temel bileşenlerle yeniden yapılandırılan imajları göster
    
    # 50'den az olan temel bileşenler kullanılmıştır
    
    if numpc <= 50:
        ax = subplot(2,3,i,frame_on=False)
        
        ax.xaxis.set_major_locator(NullLocator()) 
        ax.yaxis.set_major_locator(NullLocator())
        
        i += 1 
        
        plt.imshow(Ar)
        title('PCs # '+ str(numpc))
        gray()
        

figure()
imshow(A)
title('numpc FULL')
gray()
show()
    
    
from pylab import plot,axis

figure()

perc = cumsum(latent)/sum(latent)
dist = dist/max(dist)

plot(range(len(perc)),perc,'b',range(0,full_pc+10,10),dist,'r')
axis([0,full_pc,0,1.1])

show()

    