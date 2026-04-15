"""
Generate PDF report: Time-Series Analysis of Indian Port Activity
Reads ship_counts.csv and produces charts + narrative PDF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Style
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.0)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Load data
df = pd.read_csv('../ship_counts.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df['quarter'] = df['date'].dt.quarter
df['month_name'] = df['date'].dt.strftime('%b')

mumbai = df[df['port'] == 'mumbai'].copy()
paradip = df[df['port'] == 'paradip'].copy()

yearly = df.groupby(['port', 'year'])['ship_count'].agg(['sum', 'mean', 'count']).reset_index()
yearly.columns = ['port', 'year', 'total_ships', 'avg_monthly', 'months_observed']

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

PDF_PATH = 'Port_Activity_Analysis_Report.pdf'

with PdfPages(PDF_PATH) as pdf:

    # ── PAGE 1: Title ──
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.text(0.5, 0.70, 'Time-Series Analysis of Indian Port Activity\nand Economic Output',
            ha='center', va='center', fontsize=26, fontweight='bold',
            color='#1a237e', linespacing=1.5)
    ax.text(0.5, 0.52, 'Satellite-Based Ship Detection Study',
            ha='center', va='center', fontsize=16, color='#424242')
    ax.text(0.5, 0.42, 'Ports: Mumbai (JNPT) & Paradip  |  Period: 2018–2026',
            ha='center', va='center', fontsize=13, color='#616161')
    ax.text(0.5, 0.30, 'Methodology: YOLO11x-OBB ship detection on Sentinel-2 satellite imagery\n'
            'Source: Microsoft Planetary Computer (monthly, least-cloudy composite)',
            ha='center', va='center', fontsize=10, color='#757575', linespacing=1.8)
    ax.text(0.5, 0.15, f'Generated: {datetime.now().strftime("%B %d, %Y")}',
            ha='center', va='center', fontsize=10, color='#9e9e9e')

    # Decorative line
    ax.plot([0.2, 0.8], [0.48, 0.48], color='#1a237e', linewidth=2, alpha=0.5)

    pdf.savefig(fig)
    plt.close()

    # ── PAGE 2: Executive Summary ──
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.axis('off')

    ax.text(0.5, 0.97, 'Executive Summary', ha='center', va='top',
            fontsize=20, fontweight='bold', color='#1a237e')

    m_total = mumbai['ship_count'].sum()
    p_total = paradip['ship_count'].sum()
    m_avg = mumbai['ship_count'].mean()
    p_avg = paradip['ship_count'].mean()

    # Trend calculations
    m_sorted = mumbai.sort_values('date')
    p_sorted = paradip.sort_values('date')
    m_slope = stats.linregress(np.arange(len(m_sorted)), m_sorted['ship_count']).slope
    p_slope = stats.linregress(np.arange(len(p_sorted)), p_sorted['ship_count']).slope

    summary = (
        f"This report analyzes maritime activity at two major Indian ports — Mumbai (JNPT) "
        f"and Paradip — using satellite-based ship detection over {df['date'].min().strftime('%B %Y')} "
        f"to {df['date'].max().strftime('%B %Y')}.\n\n"
        f"Key Findings:\n\n"
        f"  • Mumbai: {int(m_total)} total ship detections across {len(mumbai)} months "
        f"(avg {m_avg:.1f}/month)\n"
        f"  • Paradip: {int(p_total)} total ship detections across {len(paradip)} months "
        f"(avg {p_avg:.1f}/month)\n"
        f"  • Mumbai trend: {'increasing' if m_slope > 0 else 'decreasing'} "
        f"({m_slope:+.3f} ships/month)\n"
        f"  • Paradip trend: {'increasing' if p_slope > 0 else 'decreasing'} "
        f"({p_slope:+.3f} ships/month)\n"
        f"  • Both ports show seasonal patterns influenced by monsoon and trade cycles\n"
        f"  • COVID-19 lockdowns (2020) caused measurable disruptions to port activity\n\n"
        f"Methodology:\n\n"
        f"  1. Sentinel-2 L2A satellite imagery obtained from Microsoft Planetary Computer\n"
        f"     (monthly, least-cloudy image per month, 2018–2026)\n"
        f"  2. Ship detection via YOLO11x-OBB model with SAHI-style tiling:\n"
        f"     — 3x bicubic upscaling for small vessel detection\n"
        f"     — 640×640 tiles with 40% overlap\n"
        f"     — Oriented Bounding Box (OBB) detection + NMS\n"
        f"     — Area outlier removal (buildings/structures)\n"
        f"  3. Monthly ship counts aggregated for time-series analysis\n"
    )

    ax.text(0.05, 0.88, summary, ha='left', va='top', fontsize=10.5,
            color='#212121', linespacing=1.6, family='sans-serif',
            transform=ax.transAxes, wrap=True)

    pdf.savefig(fig)
    plt.close()

    # ── PAGE 3: Yearly Totals ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Total Ships Detected Per Year', fontsize=16, fontweight='bold', y=1.02)

    m_yr = yearly[yearly['port'] == 'mumbai']
    bars1 = axes[0].bar(m_yr['year'], m_yr['total_ships'], color='#2196F3', edgecolor='white', width=0.7)
    axes[0].set_title('Mumbai Port', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Total Ships')
    for bar, val in zip(bars1, m_yr['total_ships']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(int(val)), ha='center', va='bottom', fontsize=8, fontweight='bold')

    p_yr = yearly[yearly['port'] == 'paradip']
    bars2 = axes[1].bar(p_yr['year'], p_yr['total_ships'], color='#FF9800', edgecolor='white', width=0.7)
    axes[1].set_title('Paradip Port', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Total Ships')
    for bar, val in zip(bars2, p_yr['total_ships']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(int(val)), ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 4: Monthly Heatmaps ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle('Ship Count Heatmap — Month × Year', fontsize=16, fontweight='bold', y=1.02)

    for ax, port, cmap, title in [
        (axes[0], mumbai, 'Blues', 'Mumbai'),
        (axes[1], paradip, 'Oranges', 'Paradip')
    ]:
        pivot = port.pivot_table(index='month', columns='year', values='ship_count', aggfunc='sum')
        pivot.index = [month_labels[m-1] for m in pivot.index]
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap, ax=ax,
                    linewidths=0.5, cbar_kws={'label': 'Ships'}, annot_kws={'size': 7})
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Month')
        ax.set_xlabel('Year')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 5: Monthly bar charts (4 years) ──
    sample_years = [2019, 2020, 2022, 2024]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Month-Wise Ship Detection — Selected Years', fontsize=16, fontweight='bold', y=1.01)
    axes = axes.flatten()

    for idx, year in enumerate(sample_years):
        ax = axes[idx]
        m_data = mumbai[mumbai['year'] == year].set_index('month')['ship_count']
        p_data = paradip[paradip['year'] == year].set_index('month')['ship_count']

        x = np.arange(1, 13)
        width = 0.35
        m_vals = [m_data.get(m, 0) for m in x]
        p_vals = [p_data.get(m, 0) for m in x]

        ax.bar(x - width/2, m_vals, width, label='Mumbai', color='#2196F3', edgecolor='white')
        ax.bar(x + width/2, p_vals, width, label='Paradip', color='#FF9800', edgecolor='white')
        ax.set_title(f'{year}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Ships')
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 6: Time-series trend ──
    fig, ax = plt.subplots(figsize=(11, 5))

    for port_name, port_df, color in [('Mumbai', mumbai, '#2196F3'), ('Paradip', paradip, '#FF9800')]:
        ps = port_df.sort_values('date')
        ax.plot(ps['date'], ps['ship_count'], alpha=0.25, color=color, linewidth=1)
        rolling = ps.set_index('date')['ship_count'].rolling(window=6, min_periods=3).mean()
        ax.plot(rolling.index, rolling.values, color=color, linewidth=2.5,
                label=f'{port_name} (6-mo avg)')

    ax.axvspan(pd.Timestamp('2020-03-25'), pd.Timestamp('2020-06-30'),
               alpha=0.15, color='red')
    ax.annotate('COVID-19\nLockdown', xy=(pd.Timestamp('2020-05-01'), ax.get_ylim()[1]*0.85),
                fontsize=9, color='red', ha='center', fontstyle='italic')

    ax.set_title('Ship Activity Trend — Mumbai vs Paradip (2018–2026)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Ships Detected per Month')
    ax.legend(fontsize=11)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 7: Seasonality ──
    fig, ax = plt.subplots(figsize=(11, 5))

    for port_name, port_df, color in [('Mumbai', mumbai, '#2196F3'), ('Paradip', paradip, '#FF9800')]:
        seasonal = port_df.groupby('month')['ship_count'].agg(['mean', 'std']).reset_index()
        ax.plot(seasonal['month'], seasonal['mean'], 'o-', color=color,
                linewidth=2, markersize=8, label=f'{port_name}')
        ax.fill_between(seasonal['month'],
                        seasonal['mean'] - seasonal['std'],
                        seasonal['mean'] + seasonal['std'],
                        alpha=0.15, color=color)

    ax.set_title('Seasonal Pattern — Average Ships by Month (All Years)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Ships Detected')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.legend(fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 8: YoY Growth ──
    fig, ax = plt.subplots(figsize=(11, 5))

    for port_name, color in [('mumbai', '#2196F3'), ('paradip', '#FF9800')]:
        p_yr = yearly[yearly['port'] == port_name].sort_values('year').copy()
        p_yr = p_yr[p_yr['months_observed'] >= 10]
        p_yr['yoy_growth'] = p_yr['total_ships'].pct_change() * 100
        p_valid = p_yr.dropna(subset=['yoy_growth'])

        offset = 0.2 if port_name == 'paradip' else -0.2
        ax.bar(p_valid['year'] + offset, p_valid['yoy_growth'], width=0.35,
               color=color, label=port_name.title(), edgecolor='white')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('Year-over-Year Growth in Ship Activity (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('YoY Growth (%)')
    ax.legend(fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 9: COVID Impact ──
    def categorize_period(year):
        if year <= 2019:
            return 'Pre-COVID (2018-19)'
        elif year <= 2021:
            return 'COVID (2020-21)'
        else:
            return 'Post-COVID (2022+)'

    df['period'] = df['year'].apply(categorize_period)
    period_order = ['Pre-COVID (2018-19)', 'COVID (2020-21)', 'Post-COVID (2022+)']
    colors_period = ['#4CAF50', '#F44336', '#2196F3']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('COVID-19 Impact on Port Activity', fontsize=16, fontweight='bold', y=1.02)

    for ax, port_name in [(axes[0], 'mumbai'), (axes[1], 'paradip')]:
        port_data = df[df['port'] == port_name]
        period_avg = port_data.groupby('period')['ship_count'].mean().reindex(period_order)
        bars = ax.bar(period_order, period_avg, color=colors_period, edgecolor='white', width=0.6)
        for bar, val in zip(bars, period_avg):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(f'{port_name.title()}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Avg Ships per Month')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 10: Port Comparison ──
    fig, ax = plt.subplots(figsize=(11, 6))

    m_yr = yearly[yearly['port'] == 'mumbai'].sort_values('year')
    p_yr = yearly[yearly['port'] == 'paradip'].sort_values('year')
    common_years = sorted(set(m_yr['year']) & set(p_yr['year']))
    m_vals = m_yr[m_yr['year'].isin(common_years)].set_index('year')['total_ships']
    p_vals = p_yr[p_yr['year'].isin(common_years)].set_index('year')['total_ships']

    x = np.arange(len(common_years))
    width = 0.35

    b1 = ax.bar(x - width/2, [m_vals.get(y, 0) for y in common_years],
                width, label='Mumbai', color='#2196F3', edgecolor='white')
    b2 = ax.bar(x + width/2, [p_vals.get(y, 0) for y in common_years],
                width, label='Paradip', color='#FF9800', edgecolor='white')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        str(int(h)), ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Mumbai vs Paradip — Yearly Ship Count Comparison', fontsize=15, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Ships Detected')
    ax.set_xticks(x)
    ax.set_xticklabels(common_years)
    ax.legend(fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 11: Ratio & Cumulative ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ratio_df = pd.DataFrame({'year': common_years})
    ratio_df['mumbai'] = [m_vals.get(y, 0) for y in common_years]
    ratio_df['paradip'] = [p_vals.get(y, 0) for y in common_years]
    ratio_df['ratio'] = ratio_df['mumbai'] / ratio_df['paradip'].replace(0, np.nan)

    axes[0].plot(ratio_df['year'], ratio_df['ratio'], 'o-', color='#9C27B0', linewidth=2, markersize=8)
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Mumbai-to-Paradip Ship Ratio', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Ratio (Mumbai / Paradip)')

    m_s = mumbai.sort_values('date').copy()
    p_s = paradip.sort_values('date').copy()
    m_s['cumulative'] = m_s['ship_count'].cumsum()
    p_s['cumulative'] = p_s['ship_count'].cumsum()

    axes[1].plot(m_s['date'], m_s['cumulative'], color='#2196F3', linewidth=2, label='Mumbai')
    axes[1].plot(p_s['date'], p_s['cumulative'], color='#FF9800', linewidth=2, label='Paradip')
    axes[1].set_title('Cumulative Ship Detections', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Ships')
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 12: Quarterly boxplot ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Quarterly Distribution of Ship Activity', fontsize=16, fontweight='bold', y=1.02)

    for ax, port_name, port_df, pal in [
        (axes[0], 'Mumbai', mumbai, 'Blues'),
        (axes[1], 'Paradip', paradip, 'Oranges')
    ]:
        sns.boxplot(data=port_df, x='quarter', y='ship_count', palette=pal, ax=ax, width=0.5)
        ax.set_title(port_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Ships Detected')
        ax.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PAGE 13: Scatter + Correlation ──
    merged = mumbai[['date', 'ship_count']].merge(
        paradip[['date', 'ship_count']], on='date', suffixes=('_mumbai', '_paradip')
    )
    if len(merged) > 5:
        corr, p_val = stats.pearsonr(merged['ship_count_mumbai'], merged['ship_count_paradip'])

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(merged['ship_count_mumbai'], merged['ship_count_paradip'],
                   alpha=0.6, s=60, color='#673AB7', edgecolors='white')
        z = np.polyfit(merged['ship_count_mumbai'], merged['ship_count_paradip'], 1)
        poly = np.poly1d(z)
        x_line = np.linspace(merged['ship_count_mumbai'].min(), merged['ship_count_mumbai'].max(), 100)
        ax.plot(x_line, poly(x_line), '--', color='gray', linewidth=1.5, alpha=0.7)
        ax.set_title(f'Mumbai vs Paradip — Monthly Correlation (r={corr:.3f}, p={p_val:.4f})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Mumbai Ships/Month')
        ax.set_ylabel('Paradip Ships/Month')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # ── PAGE 14: Statistical Summary Table ──
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.88])
    ax.axis('off')

    ax.text(0.5, 0.97, 'Statistical Summary', ha='center', va='top',
            fontsize=20, fontweight='bold', color='#1a237e')

    stat_text = ""
    for port_name, port_df in [('Mumbai', mumbai), ('Paradip', paradip)]:
        ps = port_df.sort_values('date')
        x_num = np.arange(len(ps))
        slope, intercept, r_val, pv, se = stats.linregress(x_num, ps['ship_count'])

        stat_text += f"\n{port_name}:\n"
        stat_text += f"  Total observations:    {len(port_df)} months\n"
        stat_text += f"  Total ships detected:  {port_df['ship_count'].sum():.0f}\n"
        stat_text += f"  Mean ships/month:      {port_df['ship_count'].mean():.1f}\n"
        stat_text += f"  Median ships/month:    {port_df['ship_count'].median():.1f}\n"
        stat_text += f"  Std deviation:         {port_df['ship_count'].std():.1f}\n"
        stat_text += f"  Min:  {port_df['ship_count'].min():.0f}  ({port_df.loc[port_df['ship_count'].idxmin(), 'date'].strftime('%Y-%m')})\n"
        stat_text += f"  Max:  {port_df['ship_count'].max():.0f}  ({port_df.loc[port_df['ship_count'].idxmax(), 'date'].strftime('%Y-%m')})\n"
        stat_text += f"  Trend: {'increasing' if slope > 0 else 'decreasing'} ({slope:+.3f}/month, R²={r_val**2:.3f})\n"

    if len(merged) > 5:
        stat_text += f"\nCross-Port Correlation:\n"
        stat_text += f"  Pearson r = {corr:.3f} (p = {p_val:.4f})\n"
        strength = 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak'
        stat_text += f"  Interpretation: {strength} {'positive' if corr > 0 else 'negative'} correlation\n"

    ax.text(0.05, 0.88, stat_text, ha='left', va='top', fontsize=11,
            color='#212121', family='monospace', linespacing=1.5,
            transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close()

print(f"PDF report saved: {PDF_PATH}")
