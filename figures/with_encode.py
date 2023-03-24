import pandas as pd
import matplotlib.pyplot as plt
categories = ['hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L']
values_1 = [0.94,0.95,0.95,0.96,0.94,0.93,0.88]
values_2 = [0.74,0.75,0.75,0.76,0.74,0.73,0.7]

plt.bar(categories,values_1,label='GeneSpider',color='blue')
plt.bar(categories,values_2,label='GeneSpider-NetEncoder',color='orange')

plt.legend(loc='lower right')
plt.xlabel("STRING")
plt.ylabel('AUROC')
plt.show()