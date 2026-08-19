day
        zone_colors.append('red')

bars = axes[1].bar(df['age'], df['heart_rate'], color=zone_colors, edgecolor='black')

axes[1].axhline(y=75, color='green', linestyle='--', linewidth=1)
axes[1].axhline(y=95, color='orange', linestyle='--', linewidth=1)
axes[1].set_ylabel("Age")
axes[1].set_xlabel("Heart Rate")
axes[1].set_title("Heart Rate Zones by Age\n(Green=Normal, Orange=Elevated, Red=High")

plt.tight_layout()
plt.savefig("day12_advanced_plots.png", dpi=150)
print("Saved!")