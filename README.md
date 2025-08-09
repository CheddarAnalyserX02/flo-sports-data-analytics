git# 📊 FloSports Data Analytics Portfolio Project

**Author:** Nicholas Fedha
**Role:** Data Analyst (Portfolio Project)
**Tech Stack:** Python, SQL, Pandas, Matplotlib, Prophet, Scikit-learn

---

## 🏟️ Project Overview

This end-to-end analytics project simulates the work of a **FloSports Data Analyst**, covering **customer segmentation, A/B testing, demand forecasting, and survey analysis**.
The goal: **Deliver actionable insights to improve retention, pricing strategies, forecasting accuracy, and customer satisfaction for a sports streaming subscription service.**

---

## 🚀 Business Context

FloSports operates in a competitive streaming environment where **retaining subscribers and maximizing revenue** are key.
Our analytics workflow focuses on answering four high-impact business questions:

1. **Who are our customers and how can we segment them for better targeting?**
2. **Would a discount campaign improve retention enough to offset lower revenue?**
3. **What will ticket demand look like in the next 30 days?**
4. **What drives customer satisfaction and loyalty?**

---

## 📂 Project Phases

### **Phase 1 — Data Acquisition & Preparation**

* Selected and cleaned multiple datasets (subscription data, survey data, time-series ticket sales)
* Created a unified project structure for analysis
* Handled missing values, outliers, and inconsistent date formats

---

### **Phase 2 — Customer Segmentation**

* Used **RFM (Recency, Frequency, Monetary) analysis** to segment customers
* Identified high-value subscribers and at-risk churn segments
* Business impact: Informs targeted retention campaigns and premium upsells

📌 *Visual Example:*
*(Insert RFM segmentation chart here)*

---

### **Phase 3 — A/B Testing (Price Discount Impact)**

* Simulated a 20% discount campaign for Group B
* Compared retention and ARPU between control and treatment groups
* Statistical significance testing with t-tests
* **Key Insight:** Retention rose by X%, but ARPU dropped — optimal campaign window is short-term

📌 *Visual Example:*
*(Insert retention rate comparison chart here)*

---

### **Phase 4 — Demand Forecasting**

* Aggregated hourly ticket sales data to daily level
* Forecasted **30 days ahead** using Facebook Prophet
* Measured performance with MAE & RMSE metrics
* **Key Insight:** Predicted ticket spikes align with major sports events, enabling staffing & inventory planning

📌 *Visual Example:*
*(Insert forecast vs actual plot here)*

---

### **Phase 5 — Survey Analytics**

* Analyzed **Customer Satisfaction (CSAT)** and **Net Promoter Score (NPS)** data
* Performed text analysis on open-ended feedback with word clouds and sentiment scores
* **Key Insight:** Promoters cite “coverage quality,” detractors cite “pricing & streaming issues”
* Recommended targeted quality improvements over blanket discounts

📌 *Visual Example:*
*(Insert NPS breakdown & word cloud here)*

---

## 🛠️ Tech Stack & Tools

* **Languages:** Python, SQL
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Prophet
* **Visualization:** Matplotlib, WordCloud
* **Version Control:** Git & GitHub
* **Environment:** VS Code

---

## 📌 How to Run This Project

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/flo-sports-data-analytics.git
   cd flo-sports-data-analytics
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Place datasets in the `/data` folder
4. Run individual phase scripts in `/scripts`
5. Outputs (charts & PDFs) are saved in `/output`

---

## 📢 Business Value Delivered

* Identified **churn-risk customers** for targeted retention
* Quantified **ROI of discount campaigns**
* Produced **demand forecasts** to optimize event planning
* Mapped **customer sentiment** to actionable operational changes

---

## 🔗 Links

* **LinkedIn:** \[www.linkedin.com/in/nicholas-fedha-583440377]
* **Portfolio:** \[Your Portfolio Site]
* **GitHub Repo:** *(Current link)*

