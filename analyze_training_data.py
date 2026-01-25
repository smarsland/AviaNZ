"""
Analyze and visualize training data distribution.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

with open('doc_labels.json', 'r') as f:
    data = json.load(f)

files = data['files']
categories = data['categories']

# Class frequencies
class_counts = Counter()
multi_label_counts = []
samples_per_class = {cat: 0 for cat in categories}

for file_info in files:
    class_names = file_info.get('class_names', [])
    multi_label_counts.append(len(class_names))
    
    for cls in class_names:
        class_counts[cls] += 1
        if cls in samples_per_class:
            samples_per_class[cls] += 1

sorted_classes = sorted(samples_per_class.items(), key=lambda x: x[1], reverse=True)
class_names_sorted = [x[0] for x in sorted_classes]
class_counts_sorted = [x[1] for x in sorted_classes]

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Class frequency distribution (log scale)
ax1 = plt.subplot(2, 2, 1)
bars = ax1.bar(range(len(class_counts_sorted)), class_counts_sorted, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.set_yscale('log')
ax1.set_xlabel('Class (sorted by frequency)', fontsize=12)
ax1.set_ylabel('Number of samples (log scale)', fontsize=12)
ax1.set_title(f'Class Distribution - {len(categories)} Classes, 285x Imbalance', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100 samples threshold')
ax1.legend()

# Color bars by frequency
for i, (bar, count) in enumerate(zip(bars, class_counts_sorted)):
    if count < 100:
        bar.set_color('red')
    elif count < 500:
        bar.set_color('orange')
    elif count < 1000:
        bar.set_color('gold')

# 2. Labels per sample distribution
ax2 = plt.subplot(2, 2, 2)
label_dist = Counter(multi_label_counts)
labels_x = sorted(label_dist.keys())
labels_y = [label_dist[x] for x in labels_x]
bars2 = ax2.bar(labels_x, labels_y, color='forestgreen', edgecolor='black', linewidth=1)
ax2.set_xlabel('Number of labels per sample', fontsize=12)
ax2.set_ylabel('Number of samples', fontsize=12)
ax2.set_title(f'Multi-Label Distribution - {100*sum(x>1 for x in multi_label_counts)/len(files):.1f}% Multi-Label', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(labels_x, labels_y)):
    ax2.text(x, y, f'{y}\n({100*y/len(files):.1f}%)', ha='center', va='bottom', fontsize=9)

# 3. Top 30 classes with counts
ax3 = plt.subplot(2, 2, 3)
top_n = 30
top_classes = class_names_sorted[:top_n]
top_counts = class_counts_sorted[:top_n]
y_pos = np.arange(len(top_classes))
colors = ['red' if c < 100 else 'orange' if c < 500 else 'gold' if c < 1000 else 'steelblue' for c in top_counts]
bars3 = ax3.barh(y_pos, top_counts, color=colors, edgecolor='black', linewidth=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_classes, fontsize=8)
ax3.set_xlabel('Number of samples', fontsize=12)
ax3.set_title(f'Top {top_n} Classes by Frequency', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# 4. Class imbalance summary
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""
TRAINING DATA SUMMARY
{'='*50}

Total samples: {len(files):,}
Total classes: {len(categories)}

Class Distribution:
  • Most common: {max(class_counts_sorted):,} samples ({class_names_sorted[0]})
  • Least common: {min(class_counts_sorted):,} samples ({class_names_sorted[-1]})
  • Imbalance ratio: {max(class_counts_sorted)/min(class_counts_sorted):.1f}x
  
  • Classes with <100 samples: {sum(1 for c in class_counts_sorted if c < 100)} ({100*sum(1 for c in class_counts_sorted if c < 100)/len(categories):.1f}%)
  • Classes with <500 samples: {sum(1 for c in class_counts_sorted if c < 500)} ({100*sum(1 for c in class_counts_sorted if c < 500)/len(categories):.1f}%)
  • Classes with >1000 samples: {sum(1 for c in class_counts_sorted if c > 1000)} ({100*sum(1 for c in class_counts_sorted if c > 1000)/len(categories):.1f}%)

Multi-Label Statistics:
  • Single-label samples: {sum(1 for x in multi_label_counts if x == 1):,} ({100*sum(1 for x in multi_label_counts if x == 1)/len(files):.1f}%)
  • Multi-label samples: {sum(1 for x in multi_label_counts if x > 1):,} ({100*sum(1 for x in multi_label_counts if x > 1)/len(files):.1f}%)
  • Max labels/sample: {max(multi_label_counts)}
  • Mean labels/sample: {np.mean(multi_label_counts):.2f}

Top Multi-Label Combinations:
"""

combo_counts = Counter()
for file_info in files:
    class_names = file_info.get('class_names', [])
    if len(class_names) > 1:
        combo = tuple(sorted(class_names))
        combo_counts[combo] += 1

for i, (combo, count) in enumerate(combo_counts.most_common(5)):
    summary_text += f"  {i+1}. {', '.join(combo[:2])}{'...' if len(combo) > 2 else ''}: {count}x\n"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('training_data_analysis.png', dpi=150, bbox_inches='tight')
print("Saved training_data_analysis.png")
plt.close()

# Second figure: Detailed class imbalance
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# All classes linear scale
ax1.bar(range(len(class_counts_sorted)), class_counts_sorted, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Class rank (1 = most common)', fontsize=12)
ax1.set_ylabel('Number of samples', fontsize=12)
ax1.set_title('Class Distribution - Linear Scale (Long Tail Problem)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=1000, color='green', linestyle='--', linewidth=2, label='1000 samples')
ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100 samples')
ax1.legend()

# Bottom 40 classes
ax2.barh(range(40), class_counts_sorted[-40:][::-1], color='red', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(40))
ax2.set_yticklabels(class_names_sorted[-40:][::-1], fontsize=7)
ax2.set_xlabel('Number of samples', fontsize=12)
ax2.set_title('Bottom 40 Rarest Classes (Model Struggles Here)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('class_imbalance_detail.png', dpi=150, bbox_inches='tight')
print("Saved class_imbalance_detail.png")

print("\nDONE")
