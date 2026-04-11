# AttendIQ Enterprise 🚀📍

A next-generation, geo-secured attendance management system powered by Python, Flask, and Machine Learning. 

AttendIQ modernizes the classroom experience by combining instant QR-code check-ins with HTML5 Geolocation and Scikit-Learn anomaly detection to eliminate proxy attendance and provide educators with deep, actionable analytics.

## ✨ Key Features

* **📍 Geo-Fenced Check-ins:** Students must be physically present within a 300-meter radius of the campus (configured for NIET Greater Noida) to successfully check in. 
* **🤖 ML Proxy Detection:** Utilizes `IsolationForest` to analyze device fingerprints, check-in timing, and physical distance to automatically flag suspicious (proxy) attendance.
* **📊 Predictive Analytics:** Employs `K-Means` clustering to group students into risk tiers (Elite, Average, At-Risk) and `LinearRegression` to forecast end-of-semester attendance trends.
* **📱 Mobile-First Student Portal:** A beautiful, responsive, glassmorphic web interface tailored for mobile devices, complete with particle animations and real-time GPS feedback.
* **📈 Real-Time Dashboard:** A modern admin panel featuring live telemetry, heatmap density charts, and beautifully rendered data visualizations using Chart.js.
* **🗄️ Zero-Config Database:** Built-in SQLite database with automatic dummy-data seeding for instant testing.

## 🛠️ Tech Stack

* **Backend:** Python 3, Flask, SQLite3
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Frontend:** HTML5, CSS3 (Tailwind-inspired custom UI), Vanilla JavaScript
* **Utilities:** `qrcode` (Gate generation), `cryptography` (Local SSL), HTML5 Geolocation API

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/AttendIQ.git](https://github.com/yourusername/AttendIQ.git)
cd AttendIQ
