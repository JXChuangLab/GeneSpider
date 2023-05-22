import pandas as pd
import matplotlib.pyplot as plt
categories = ['hESC','hHEP','mDC','mESC','mHSC-E','mHSC-GM','mHSC-L']

bench_mark =pd.read_csv("./encode/benchmark_balance.csv",sep=',',header=None)
netcode= pd.read_csv("./encode/encode.csv",sep=',',header=None)
for name in ['STRING',"Non-Specific",'Specific']:
    cat = bench_mark[bench_mark[0]==name]
    net_values = netcode[netcode[0]==name]

    plt.figure()
    net_values_500 = net_values[net_values[2]==500][6].values
    gasf_values_500 = cat[cat[2]==500][6].values
    x = range(0,len(categories))
    plt.xticks(x,categories)
    plt.bar(x, gasf_values_500, label='GeneSpider', color=['#b07f59'])
    plt.bar(x, net_values_500, label='GeneSpider-NetCode', color=['#e7d1c1'])
    plt.legend(loc='lower right')
    if name == 'STRING':
        plt.xlabel("STRING_500")
    if name == 'Non-Specific':
        plt.xlabel("Non-specific_500")
    if name == "Specific":
        plt.xlabel("Cell-type-specific_500")
    for i,v_dir,v_bid in zip(x,net_values_500,gasf_values_500):
        str = "{:.2f}/{:.2f}".format(v_bid,v_dir)
        h = max(v_bid,v_dir)
        plt.text(i,h,str,ha='center',va='bottom')
    plt.ylabel('AUPRC')
    plt.show()

    plt.figure()
    net_values_1000 = net_values[net_values[2] == 1000][6].values
    gasf_values_1000 = cat[cat[2] == 1000][6].values
    x = range(0, len(categories))
    plt.xticks(x, categories)
    plt.bar(x, gasf_values_1000, label='GeneSpider', color=['#b07f59'])
    plt.bar(x, net_values_1000, label='GeneSpider-NetCode', color=['#e7d1c1'])
    plt.legend(loc='lower right')
    if name == 'STRING':
        plt.xlabel("STRING_1000")
    if name == 'Non-Specific':
        plt.xlabel("Non-specific_1000")
    if name == "Specific":
        plt.xlabel("Cell-type-specific_1000")
    for i, v_dir, v_bid in zip(x, net_values_1000, gasf_values_1000):
        str = "{:.2f}/{:.2f}".format(v_bid, v_dir)
        h = max(v_bid, v_dir)
        plt.text(i, h, str, ha='center', va='bottom')
    plt.ylabel('AUPRC')
    plt.show()





