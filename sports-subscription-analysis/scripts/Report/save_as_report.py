# save_as_report.py
# Run with: python save_as_report.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Config / paths
# -----------------------------
CSV_PATH = "flo_sports_customers.csv"            # update if needed
OUTPUT_PDF = "Customer_Segmentation_Report.pdf"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
# ensure columns expected exist
required_cols = {"Age", "TenureMonths", "AvgMonthlyWatchHours", "Churned"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"The CSV is missing required columns: {missing}")

# -----------------------------
# 2) Standardize & KMeans
# -----------------------------
features = ["Age", "TenureMonths", "AvgMonthlyWatchHours"]
X = df[features].fillna(0)   # basic fill; adjust if you have different NA strategy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for plotting
pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_res[:,0], pca_res[:,1]

# -----------------------------
# 3) Attach human-readable names
#    (If you'd rather auto-generate names we can do that; here we use the mapping from Notebook 2)
# -----------------------------
segment_names = {
    0: "Loyal Heavy Users",
    1: "At-Risk Mid-Tenure",
    2: "New Low-Engagement Users",
    3: "Young High-Engagement"
}
# If cluster labels differ in your run, you can reassign mapping after inspecting cluster_profile
df["SegmentName"] = df["Cluster"].map(segment_names)

# -----------------------------
# 4) Create & save charts
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150})

# 4.1 Churn rate by segment
churn_by_segment = df.groupby("SegmentName")["Churned"].mean().sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=churn_by_segment.index, y=churn_by_segment.values)
plt.xticks(rotation=25)
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Segment")
plt.tight_layout()
chart1 = os.path.join(CHARTS_DIR, "churn_by_segment.png")
plt.savefig(chart1)
plt.close()

# 4.2 Average Tenure / Age / Watch Hours by segment
metrics = ["TenureMonths", "Age", "AvgMonthlyWatchHours"]
chart_paths = [chart1]
for metric in metrics:
    plt.figure(figsize=(6,4))
    order = df.groupby("SegmentName")[metric].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="SegmentName", y=metric, estimator=np.mean, order=order)
    plt.xticks(rotation=25)
    plt.title(f"Average {metric} by Segment")
    plt.tight_layout()
    p = os.path.join(CHARTS_DIR, f"{metric}_by_segment.png")
    plt.savefig(p)
    plt.close()
    chart_paths.append(p)

# 4.3 PCA scatter colored by segment
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="SegmentName", s=40)
plt.title("Customer Segments (PCA projection)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
pca_chart = os.path.join(CHARTS_DIR, "pca_segments.png")
plt.savefig(pca_chart)
plt.close()
chart_paths.append(pca_chart)

# -----------------------------
# 5) Segment summary table (for PDF)
# -----------------------------
segment_summary = df.groupby("SegmentName").agg(
    Customers=("SegmentName", "count"),
    AvgAge=("Age", "mean"),
    AvgTenure=("TenureMonths", "mean"),
    AvgWatchHours=("AvgMonthlyWatchHours", "mean"),
    ChurnRate=("Churned", "mean")
).reset_index()

# Format numeric values for display
segment_summary_display = segment_summary.copy()
segment_summary_display["AvgAge"] = segment_summary_display["AvgAge"].round(1)
segment_summary_display["AvgTenure"] = segment_summary_display["AvgTenure"].round(1)
segment_summary_display["AvgWatchHours"] = segment_summary_display["AvgWatchHours"].round(1)
segment_summary_display["ChurnRate"] = (segment_summary_display["ChurnRate"] * 100).round(1).astype(str) + "%"

# -----------------------------
# 6) Build PDF with ReportLab
# -----------------------------
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
story = []

# Title
story.append(Paragraph("FloSports — Customer Segmentation Report", styles["Title"]))
story.append(Spacer(1, 8))

# Intro
intro = ("This report contains the segmentation analysis based on Age, Tenure (months), and Average Monthly Watch Hours. "
         "K-Means (k=4) was used to generate customer segments. Charts below visualize segment differences and churn risk.")
story.append(Paragraph(intro, styles["Normal"]))
story.append(Spacer(1, 12))

# Insights (short)
insights = [
    ("Loyal Heavy Users", "Long tenure, high watch hours, lowest churn — recommend upsell and loyalty programs."),
    ("At-Risk Mid-Tenure", "Mid tenure with moderate engagement and elevated churn — prioritize re-engagement campaigns."),
    ("New Low-Engagement Users", "Short tenure and low watch hours — strengthen onboarding, highlight core value quickly."),
    ("Young High-Engagement", "Younger, engaged users — promote interactive features and social sharing.")
]
story.append(Paragraph("<b>Segment Insights (summary)</b>", styles["Heading2"]))
for seg, text in insights:
    story.append(Paragraph(f"<b>{seg}:</b> {text}", styles["Normal"]))
    story.append(Spacer(1,6))

story.append(Spacer(1,10))

# Charts
story.append(Paragraph("<b>Charts</b>", styles["Heading2"]))
for cp in chart_paths:
    if os.path.exists(cp):
        # ReportLab Image size: adjust if needed
        story.append(Image(cp, width=400, height=250))
        story.append(Spacer(1,8))

# Segment summary table
story.append(Spacer(1,10))
story.append(Paragraph("<b>Segment Summary Table</b>", styles["Heading2"]))
table_data = [["Segment", "Customers", "Avg Age", "Avg Tenure (mo)", "Avg Watch Hrs", "Churn Rate"]]
for _, row in segment_summary_display.iterrows():
    table_data.append([
        row["SegmentName"],
        int(row["Customers"]),
        row["AvgAge"],
        row["AvgTenure"],
        row["AvgWatchHours"],
        row["ChurnRate"]
    ])

table = Table(table_data, hAlign="LEFT")
table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.grey),
    ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
    ("ALIGN",(1,1),(-1,-1),"CENTER"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ("BACKGROUND",(0,1),(-1,-1), colors.whitesmoke)
]))
story.append(table)

# Footer note
story.append(Spacer(1,12))
story.append(Paragraph("Generated by Notebook 2 pipeline. CSV used: " + CSV_PATH, styles["Italic"]))

# Build PDF
doc.build(story)
print(f"PDF saved as: {OUTPUT_PDF}")
print("Charts saved to:", os.path.abspath(CHARTS_DIR))
