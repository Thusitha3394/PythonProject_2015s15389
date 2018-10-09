
# coding: utf-8

# In[4]:

from pylab import*

t=arange(0.0,20.0,1)
s=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
s2=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
plot(t,s)
plot(t,s2)

xlabel('item(s)')
ylabel('value')
title('Python Line Chart: Plotting numbers')
grid(True)
show()


# In[6]:

import matplotlib.pyplot as plt;plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects=('Python','C++','Java','Perl','Scala','Lisp')
y_pos=np.arange(len(objects))
performance = [10,8,6,4,2,1]

plt.bar(y_pos,performance,align='center',alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Usage')
plt.title('Programming language usage')

plt.show()


# In[7]:

import numpy as np
import matplotlib.pyplot as plt

#data to plot
n_groups = 4
means_frank = (90,55,40,65)
means_guido = (85,62,54,20)

#create plot
fig,ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1=plt.bar(index,means_frank,bar_width,
              alpha=opacity,
              color='b',
              label='Frank')

rects2 = plt.bar(index+bar_width,means_guido,bar_width,
                alpha=opacity,
                color='g',
                label='Guido')
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width,('A','B','C','D'))
plt.legend()

plt.tight_layout()
plt.show()


# In[10]:

import matplotlib.pyplot as plt

#Data to plot
labels = 'Python','C++','Ruby','Java'
sizes = [215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode = (0.1,0,0,0)#explode 1st slice

#plot
plt.pie(sizes,explode=explode,labels=labels,colors=colors,
       autopct='%1.1f%%',shadow=True,startangle=140)

plt.axis('equal')
plt.show()

#To add a legend,use plt.legend function
import matplotlib.pyplot as plt

labels = ['Cookies','Jellybean','Milkshake','Cheesecake']
sizes = [38.4,40.6,20.7,10.3]
colors = ['yellowgreen','gold','lightskyblue','lightcoral']
patches,texts=plt.pie(sizes,colors=colors,shadow=True,startangle=90)
plt.legend(patches,labels,loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[11]:

import matplotlib.pyplot as plt
import numpy as np

y=[2,4,6,8,10,12,14,16,18,20]
y2=[10,11,12,13,14,15,16,17,18,19]
x=np.arange(10)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,y,label='$y=numbers')
ax.plot(x,y2,label='$y2=other numbers')
plt.title('Legend inside')
ax.legend()
plt.show()


# In[16]:

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


x=[21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
num_bins =5
n,bins,patches = plt.hist(x,num_bins,facecolor='blue',alpha=0.5)
plt.show()


# In[17]:

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#example data
mu = 100 #mean of distribution
sigma = 15 #standard deviation of distribution
x=mu + sigma*np.random.randn(1000)

num_bins=20
#the histogram of the data
n,bins,patches=plt.hist(x,num_bins,normed=1,facecolor='blue',alpha=0.5)

#add a best fit line
y=mlab.normpdf(bins,mu,sigma)
plt.plot(bins,y,'r--')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'histogram of IQ:$\mu=100$,$\sigma=15$')

#Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

#save figure
fig.savefig('plot.png')


# In[18]:

#x and y are the two variables
heatmap,xedges,yedges=np.histogram2d(x,y,bins=(64,64))
extent =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
plt.clf()
plt.title('Pythonspot.com heatmap example')
plt.ylabel('y')
plt.xlabel('x')
plt.imshow(heatmap,extent=extent)
plt.show()


# In[ ]:



