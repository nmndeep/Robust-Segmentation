import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from palettable.tableau import GreenOrange_12, TableauMedium_10, Tableau_20, ColorBlind_10 
from palettable.cartocolors.qualitative import Bold_8, Prism_10, Vivid_10, Safe_10
plt.rcParams["font.family"] = "Times New Roman"

new_colors = np.vstack((ColorBlind_10.mpl_colors, Bold_8.mpl_colors))
my_cmap = ListedColormap(new_colors, name='BoldBlind')
my_cmap2 = ListedColormap(Tableau_20.mpl_colors)
# mcmap = my_cmap1 +    my_cmap2
# sns.set_style("darkgrid")
sns.set_style("whitegrid", {"grid.color": ".8", "grid.linestyle": "--"})
sns.despine(left=True)

models = ["pgd_5step", "apgd_5step"] #, "vit_s_cvst_25ep_final3", "vit_s_cvst_25ep_convstem_high_lr"] #, "conviso_cvblk_300AT", "conviso_300AT"] #, "base_cvblk"] #, "tb10_model4", "tb10_model5", "model1"]
# l2s = [model0, model3, model5, model6, model7, model8]
train_stats = []
for m in models:
    print(m)
    with open(f"/Users/nmndeep/Documents/logs_semseg/{m}_log.txt", 'r') as fp:
        # lines to read
        # if "300" in m:
        line_numbers = np.arange(0,50,1)
        # else:
        #     line_numbers = np.arange(0,49,2)
        cnvnxt = []
        for i, line in enumerate(fp):
            # read line 4 and 7
            print(line)
            if i in line_numbers:
                i_0 = (line.index("Loss"))
                i_1 = (line.index("Cost"))
                cnvnxt.append(float(line[i_0+6:i_1-6]))
        train_stats.append(cnvnxt)
print(train_stats)

fig = plt.figure(figsize=(12,8))
plt.title("Training curves")
# xx1 = [1]
# xx = ["Rand-init", "1k-init", "3-Aug", "RandAug+CM+MU+LS", "CvBlk", "Long-Train", "Base"]
xx1 = (list(np.arange(1,51,1)))

plt.plot(xx1[4:], train_stats[0][4:], linewidth=1.5, color = my_cmap2(12), label=models[0])
plt.plot(xx1[4:], train_stats[1][4:], linewidth=1.5, color = my_cmap2(0), label=models[1])

# plt.annotate('lr-peak', xy=(xx2[65], train_stats[0][65]+.1))
plt.title("UperNet-ConvNeXt-T for PASCALVOC-AUG")
# # plt.plot(xx1[10:49], cnvnxte4[10:], linewidth=1.5, color = my_cmap(11), linestyle ="--", label="convnext-b-1e-3_epoch10onwards")
# # plt.plot(xx1[10:46], cnvnxtcvb4[10:], linewidth=1.5, color = my_cmap(0), linestyle ="--",  label="convnext-b-cvblk1e-3_epoch10onwards")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Train Loss", fontsize=12)
plt.legend(fontsize=12)
plt.show()