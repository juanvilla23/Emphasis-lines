import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
plt.axis([0, 6, 0, 20])
plt.show()

plt.figure(1)
plt.subplot(211)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
plt.figure(1)
plt.subplot(212)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bs')
plt.axis([0, 6, 0, 20])
plt.show()


plt.figure(1)
plt.plot([2],[4],'ro')
plt.axis([0, 6, 0, 20])
plt.text(2.1, 4, 'text 0', size=8)
plt.show()


