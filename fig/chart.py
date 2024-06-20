"""
@FileName：chart.py\n
@Description：\n
@Author：WBobby\n
@Department：CUG\n
@Time：2023/12/18 23:51\n
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig_name = 'Sample Number of PPF '
figsize = (8, 8)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman'] + matplotlib.rcParams['font.serif']
# (a) bypass diode heating, (b) substring disconnection, (c) debris covering, (d) panel breaking, (e) dusty covering,
# (f) General hot spot, (g) health panel.
fault_dic = {'a': 595, 'b': 427, 'c': 71, 'd': 410, 'e': 303, 'f': 377, 'g': 131,
             'h': 800, 'i': 946, 'j': 1519}

fault_names = list(fault_dic.keys())[::]
fault_counts = list(fault_dic.values())[::]


# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
colors = plt.cm.Wistia(np.linspace(1, 0.1, len(fault_names)))
print(colors)

fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.6
bars = ax.bar(fault_names, fault_counts, color=colors, width=bar_width)
# bars = ax.bar(fault_counts, fault_names, color=colors, height=bar_width)

# plt.barh(fault_names, fault_counts, color=colors)
plt.ylabel('Fault Count', fontsize=12, fontweight='bold')
plt.xlabel('Fault Types', fontsize=12, fontweight='bold')
# plt.title(fig_name, fontsize=25)

for bar, x, count in zip(bars, fault_names, fault_counts):
    ax.text(x, bar.get_y() + bar.get_height(), str(count), ha='center', va='top', fontsize=14)

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_title(fig_name, fontsize=15, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.tight_layout()  # 调整布局
plt.savefig(r'C:\Users\Wbobby\Documents\TeX_files\PVFC\tu/' + 'Sample3.png', dpi=300)
plt.show()
