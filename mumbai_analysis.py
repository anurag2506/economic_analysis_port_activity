"""
Mumbai Port Ship Activity Analysis — generates all charts and saves them.
Reads detection .txt files, builds CSV, produces plots.
"""

import os
import glob
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style ──
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (14, 6)

# ── Build CSV from .txt detection files ──
IMG_DIR = "../satellite_imgs2/mumbai"
CSV_PATH = "../ship_counts_mumbai.csv"

rows = []
images = sorted(glob.glob(os.path.join(IMG_DIR, "mumbai_*.png")))
images = [i for i in images if "_detections" not in i]

for img_path in images:
    fname = os.path.basename(img_path)
    match = re.match(r"mumbai_(\d{4})_(\d{2})\.png", fname)
    if not match:
        continue
    year, month = int(match.group(1)), int(match.group(2))
    txt_path = img_path.replace(".png", ".txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            count = sum(1 for line in f if line.strip())
    else:
        count = 0
    rows.append({"port": "mumbai", "year": year, "month": month, "ship_count": count, "image_file": fname})

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["port", "year", "month", "ship_count", "image_file"])
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV saved: {CSV_PATH} ({len(rows)} rows)")

# ── Load into DataFrame ──
df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df['quarter'] = df['date'].dt.quarter
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ── Yearly aggregation ──
yearly = df.groupby('year')['ship_count'].agg(['sum', 'mean', 'count']).reset_index()
yearly.columns = ['year', 'total_ships', 'avg_monthly', 'months_observed']

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════
# PLOT A: Total Ships Detected Per Year
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(yearly)))
bars = ax.bar(yearly['year'], yearly['total_ships'], color=colors, edgecolor='white', width=0.7)

for bar, val, n_months in zip(bars, yearly['total_ships'], yearly['months_observed']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{int(val)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    if n_months < 12:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"({n_months}mo)", ha='center', va='center', fontsize=8, color='white', fontstyle='italic')

ax.set_title('Mumbai Port — Total Ships Detected Per Year (Satellite-Based)', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('Total Ships Detected', fontsize=13)
ax.set_xticks(yearly['year'])
ax.set_ylim(bottom=0)

# Trend line
z = np.polyfit(yearly['year'], yearly['total_ships'], 1)
p = np.poly1d(z)
ax.plot(yearly['year'], p(yearly['year']), '--', color='#D32F2F', linewidth=2, alpha=0.7, label=f'Trend ({z[0]:+.1f}/yr)')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'a_yearly_totals.png'), bbox_inches='tight')
plt.show()
print("Plot A saved: a_yearly_totals.png")

# ════════════════════════════════════════════════════
# PLOT B: Month-Wise Ships for 4 Selected Years
# ════════════════════════════════════════════════════
sample_years = [2019, 2021, 2023, 2025]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
year_colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']

for idx, (year, color) in enumerate(zip(sample_years, year_colors)):
    ax = axes[idx]
    year_data = df[df['year'] == year].set_index('month')['ship_count']
    x = np.arange(1, 13)
    vals = [year_data.get(m, 0) for m in x]

    bars = ax.bar(x, vals, color=color, edgecolor='white', width=0.7, alpha=0.85)

    # Value labels
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(int(val)), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Average line
    avg = np.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else 0
    ax.axhline(y=avg, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(12.5, avg, f'avg: {avg:.1f}', fontsize=8, color='red', va='center')

    ax.set_title(f'{year} — Monthly Ship Counts', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Ships Detected')
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylim(bottom=0)

plt.suptitle('Mumbai Port — Month-Wise Ship Detection (4 Selected Years)',
             fontsize=17, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'b_monthly_4years.png'), bbox_inches='tight')
plt.show()
print("Plot B saved: b_monthly_4years.png")

# ════════════════════════════════════════════════════
# PLOT C1: Time-Series Trend with Rolling Average
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))

df_sorted = df.sort_values('date')
ax.plot(df_sorted['date'], df_sorted['ship_count'], 'o-',
        alpha=0.3, color='#1976D2', linewidth=1, markersize=3, label='Monthly count')

# 6-month rolling
rolling = df_sorted.set_index('date')['ship_count'].rolling(window=6, min_periods=3).mean()
ax.plot(rolling.index, rolling.values, color='#D32F2F', linewidth=2.5, label='6-month moving avg')

# 12-month rolling
rolling12 = df_sorted.set_index('date')['ship_count'].rolling(window=12, min_periods=6).mean()
ax.plot(rolling12.index, rolling12.values, color='#388E3C', linewidth=2, linestyle='--', label='12-month moving avg')

# COVID shading
ax.axvspan(pd.Timestamp('2020-03-25'), pd.Timestamp('2020-06-30'),
           alpha=0.15, color='red')
ax.annotate('COVID-19\nLockdown', xy=(pd.Timestamp('2020-05-01'), ax.get_ylim()[1]*0.9),
            fontsize=9, color='red', ha='center', fontstyle='italic')

ax.set_title('Mumbai Port — Ship Activity Trend (2018–2026)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Ships Detected per Month', fontsize=13)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c1_trend_line.png'), bbox_inches='tight')
plt.show()
print("Plot C1 saved: c1_trend_line.png")

# ════════════════════════════════════════════════════
# PLOT C2: Heatmap — Month × Year
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

pivot = df.pivot_table(index='month', columns='year', values='ship_count', aggfunc='sum')
pivot.index = [month_labels[m-1] for m in pivot.index]

sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Ships Detected'}, annot_kws={'size': 9})
ax.set_title('Mumbai Port — Ship Count Heatmap (Month × Year)', fontsize=15, fontweight='bold')
ax.set_ylabel('Month', fontsize=12)
ax.set_xlabel('Year', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c2_heatmap.png'), bbox_inches='tight')
plt.show()
print("Plot C2 saved: c2_heatmap.png")

# ════════════════════════════════════════════════════
# PLOT C3: Seasonality — Average by Month
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

seasonal = df.groupby('month')['ship_count'].agg(['mean', 'std', 'median']).reset_index()

ax.plot(seasonal['month'], seasonal['mean'], 'o-', color='#1565C0',
        linewidth=2.5, markersize=10, label='Mean', zorder=3)
ax.fill_between(seasonal['month'],
                seasonal['mean'] - seasonal['std'],
                seasonal['mean'] + seasonal['std'],
                alpha=0.15, color='#1565C0')
ax.plot(seasonal['month'], seasonal['median'], 's--', color='#FF6F00',
        linewidth=2, markersize=7, label='Median', alpha=0.8)

ax.set_title('Mumbai Port — Seasonal Pattern (Average Ships by Month, All Years)',
             fontsize=15, fontweight='bold')
ax.set_xlabel('Month', fontsize=13)
ax.set_ylabel('Ships Detected', fontsize=13)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c3_seasonality.png'), bbox_inches='tight')
plt.show()
print("Plot C3 saved: c3_seasonality.png")

# ════════════════════════════════════════════════════
# PLOT C4: Year-over-Year Growth
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

full_years = yearly[yearly['months_observed'] >= 10].copy()
full_years['yoy_growth'] = full_years['total_ships'].pct_change() * 100
yoy = full_years.dropna(subset=['yoy_growth'])

colors_yoy = ['#4CAF50' if v >= 0 else '#F44336' for v in yoy['yoy_growth']]
bars = ax.bar(yoy['year'], yoy['yoy_growth'], color=colors_yoy, edgecolor='white', width=0.6)

for bar, val in zip(bars, yoy['yoy_growth']):
    offset = 1 if val >= 0 else -3
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{val:+.1f}%", ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=10, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_title('Mumbai Port — Year-over-Year Growth in Ship Activity (%)',
             fontsize=15, fontweight='bold')
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('YoY Growth (%)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c4_yoy_growth.png'), bbox_inches='tight')
plt.show()
print("Plot C4 saved: c4_yoy_growth.png")

# ════════════════════════════════════════════════════
# PLOT C5: COVID-19 Impact
# ════════════════════════════════════════════════════
def categorize_period(year):
    if year <= 2019:
        return 'Pre-COVID\n(2018-19)'
    elif year <= 2021:
        return 'COVID\n(2020-21)'
    else:
        return 'Post-COVID\n(2022+)'

df['period'] = df['year'].apply(categorize_period)
period_order = ['Pre-COVID\n(2018-19)', 'COVID\n(2020-21)', 'Post-COVID\n(2022+)']
period_colors = ['#4CAF50', '#F44336', '#2196F3']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Average monthly
period_avg = df.groupby('period')['ship_count'].mean().reindex(period_order)
bars = axes[0].bar(period_order, period_avg, color=period_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, period_avg):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_title('Avg Monthly Ships by Period', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Avg Ships per Month', fontsize=12)

# Total per period
period_total = df.groupby('period')['ship_count'].sum().reindex(period_order)
bars2 = axes[1].bar(period_order, period_total, color=period_colors, edgecolor='white', width=0.6)
for bar, val in zip(bars2, period_total):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[1].set_title('Total Ships by Period', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Ships Detected', fontsize=12)

plt.suptitle('Mumbai Port — COVID-19 Impact on Ship Activity', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c5_covid_impact.png'), bbox_inches='tight')
plt.show()
print("Plot C5 saved: c5_covid_impact.png")

# ════════════════════════════════════════════════════
# PLOT C6: Quarterly Boxplot
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='quarter', y='ship_count', palette='Blues', ax=ax, width=0.5)
ax.set_title('Mumbai Port — Ship Count Distribution by Quarter', fontsize=15, fontweight='bold')
ax.set_xlabel('Quarter', fontsize=13)
ax.set_ylabel('Ships Detected', fontsize=13)
ax.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'c6_quarterly_box.png'), bbox_inches='tight')
plt.show()
print("Plot C6 saved: c6_quarterly_box.png")

# ════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MUMBAI PORT — STATISTICAL SUMMARY")
print("=" * 60)
print(f"  Total observations:    {len(df)} months")
print(f"  Date range:            {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
print(f"  Total ships detected:  {df['ship_count'].sum()}")
print(f"  Mean ships/month:      {df['ship_count'].mean():.1f}")
print(f"  Median ships/month:    {df['ship_count'].median():.1f}")
print(f"  Std deviation:         {df['ship_count'].std():.1f}")
print(f"  Min:  {df['ship_count'].min()} ({df.loc[df['ship_count'].idxmin(), 'date'].strftime('%Y-%m')})")
print(f"  Max:  {df['ship_count'].max()} ({df.loc[df['ship_count'].idxmax(), 'date'].strftime('%Y-%m')})")

# Trend
x_num = np.arange(len(df.sort_values('date')))
slope, intercept, r_val, p_val, se = stats.linregress(x_num, df.sort_values('date')['ship_count'])
print(f"  Trend: {'increasing' if slope > 0 else 'decreasing'} ({slope:+.3f}/month, R²={r_val**2:.3f}, p={p_val:.4f})")

print("\n  Yearly breakdown:")
for _, row in yearly.iterrows():
    print(f"    {int(row['year'])}: {int(row['total_ships']):>4} ships ({int(row['months_observed'])} months, avg {row['avg_monthly']:.1f}/mo)")

print(f"\nAll plots saved to {OUT_DIR}/")
