import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
marker_list = ['o', 'v', '<', 'x']


plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

cur = 0
lens = 1568
encoding_array_rain = np.load('./t-sne-weights/zz-rain.npy', allow_pickle=True)
encoding_array_snow = np.load('./t-sne-weights/zz-snow.npy', allow_pickle=True)
encoding_array_fog = np.load('./t-sne-weights/zz-fog.npy', allow_pickle=True)
encoding_array_night = np.load('./t-sne-weights/zz-night.npy', allow_pickle=True)

encoding_array = np.concatenate((encoding_array_rain, encoding_array_snow, encoding_array_fog, encoding_array_night), 0)

class_list = ['rain', 'snow', 'fog', 'night']

n_class = len(class_list) # 测试集标签类别数
palette = sns.hls_palette(n_class) # 配色方案


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, n_iter=250)
X_tsne_2d = tsne.fit_transform(encoding_array)
print(X_tsne_2d.shape)


plt.figure(figsize=(14, 14))
for idx, fruit in enumerate(class_list): # 遍历每个类别
    # 获取颜色和点型
    color = palette[idx]
    marker = marker_list[idx%len(marker_list)]

    # 找到所有标注类别为当前类别的图像索引号
    print(cur, "   ", lens)
    plt.scatter(X_tsne_2d[cur:lens, 0], X_tsne_2d[cur:lens, 1], color=color, marker=marker, label=fruit, s=150)
    cur += 1568
    lens += 1568

plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
plt.xticks([])
plt.yticks([])
# plt.savefig('语义特征t-SNE二维降维可视化.pdf', dpi=300) # 保存图像
plt.show()