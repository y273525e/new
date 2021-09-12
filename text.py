# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 10:38:20 2021

@author: ricky
"""

"""intro"""
import matplotlib.pyplot as plt
import numpy as np
# default generate x sequence as the same length but start from 0 : [0,1,2,3]
plt.plot([1,2,3,4])
plt.ylabel("some things")
# 用plt.show可將當前結果輸出
plt.show()

"""Formatting the style of your plot"""
# x sequence [1,2,3,4], y sequence [1,4,9,16]
plt.plot([1,2,3,4],[1,4,9,16],'ro')
# axis : [xmin, xmax, ymin, ymax]
plt.axis([0,6,0,20])
plt.show


# np.arange(min, max, interval), sequence
t = np.arange(0,5,0.2)
# red dashes, blue squares, green triangles
plt.plot(t,t,'r--', t,t**2,'bs', t,t**3,'g^')
plt.show()


"""Plotting with keyword strings"""
# np.random.randint(min, max, n), random
# np.random.randn 隨機產生數字
data = {
        'a':np.arange(50),
        'c':np.random.randint(0,50,50),
        'd':np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

# c=color, s=shape
plt.scatter('a','b',c='c',s='d',data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

"""Plotting with categorical variables"""
# subplot : https://vimsky.com/zh-tw/examples/usage/matplotlib-pyplot-subplots-in-python.html
names = ['group_a','group_b','group_c']
values = [1,10,100]
plt.figure(figsize=(9, 3))
fig, axs = plt.subplots(2,2)
axs[0,0].plot(names, values)
axs[1,0].scatter(names, values)
axs[1,1].bar(names, values)
plt.suptitle('Categorical Plotting')
plt.show()


"""Controlling line properties"""
plt.plot('a','b',data=data, linewidth=2)    
plt.plot('a','b',data=data, linewidth=5)

# unpacking
# https://www.learncodewithmike.com/2019/12/python-unpacking.html
# https://stackoverflow.com/questions/43222586/python-trailing-comma-in-object-name
line, = plt.plot(t,t,'--')
line.set_antialiased(False)

lines = plt.plot(t,t,'r',t,t**2,'b')
# plt.setp對plot內的物件統一調整用
plt.setp(lines, linewidth=2.0)

"""Working with multiple figures and axes"""
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0,5,0.1)
t2 = np.arange(0,5,0.02)
plt.figure()

plt.subplot(331)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')

plt.subplot(335)
plt.plot(t2, np.cos(2*np.pi*t2),'r--')

plt.subplot(339)
plt.plot(t1,f(t1),'k',t2,f(t2),'bo')

plt.subplot(334)
plt.plot(t2, np.cos(2*np.pi*t2),'r')
plt.show

# 切換不同plot之間的操作
plt.figure(1)
plt.title('test')

plt.figure(2)
plt.plot(t2,f(t2))

plt.figure(1)
plt.scatter(t2,t2)


"""Working with text"""
mu, sigma = 100,15
x = mu + sigma * np.random.randn(10000)
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.xlabel("Smarts")
plt.ylabel("Probability")
plt.title("Histogram of IQ")
# \ 會把後面加上的東西變成符號
# 下列三種情況
plt.text(60,0.025,r'$mu=100, sigma=15$')
plt.text(60,0.025,r'$\mu=100,\sigma=15$')
plt.text(60,0.025,r'$\mu=100,\ \sigma=15$')

# example
fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)

# Set titles for the figure and the subplot respectively
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax.set_title('axes title')
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')
# Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
ax.axis([0, 10, 0, 10])
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
ax.text(3, 2, 'unicode: Institut für Festkörperphysik')
ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()


"""Logarithmic and other nonlinear axes"""
# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()


#shiny科普圖(月亮)
fig = plt.figure()
ax = fig.add_subplot()
ax.annotate('V*\u0394T', xy=(0,2), xytext=(0.5, 2),
            arrowprops=dict(facecolor='green', shrink=0.1))
ax.annotate('d', xy=(0.25,0), xytext=(0.25, 0.5),
            arrowprops=dict(facecolor='red', shrink=0.1))
ax.annotate('\u0394\u03F4', xy=(2.5,0.3), xytext=(1.9, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.1))
ax.annotate('\u03F4', xy=(2.3,0.1), xytext=(1.5, 0.3),
            arrowprops=dict(facecolor='green', shrink=0.1))
x=[0,3]
y=[3,0]
plt.plot(x, y)
x=[0,3]
y=[1,0]
plt.plot(x, y)
x=[0,0]
y=[3,1]
plt.plot(x, y)
x=[0,3]
y=[0,0]
plt.plot(x, y,'--')
