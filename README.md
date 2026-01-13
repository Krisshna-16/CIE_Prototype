# 🧠 Collective Intelligence Engine (CIE)

**Version:** 1.0 | **Domain:** AI-driven Problem Solving  

---

## Overview
The **Collective Intelligence Engine** helps solve real-world problems by **reusing proven solution patterns**.  
It uses **NLP embeddings + FAISS similarity search + visual problem decomposition** to analyze a problem and recommend actionable steps with confidence scores.

---

## Key Features
- **Problem Decomposition:** Automatically identifies dimensions like Scalability, Resource Allocation, Cost Optimization.  
- **Pattern Matching:** Finds top-matching solution patterns from a knowledge base.  
- **Ranked Recommendations:** Shows solutions with confidence and recommended steps.  
- **Explainable Reasoning:** Highlights why each solution is suggested.  
- **Visual Decomposition:** Maps problem → dimensions → solutions for clarity.

---

## Demo Problems
1. **FinTech Scaling:** Platform crashes during high traffic. How to scale efficiently without huge costs?  
2. **Smart City Traffic:** Rush-hour congestion delays emergency vehicles. Optimize traffic flow.  
3. **Manufacturing Efficiency:** Idle and overloaded production lines. Reduce errors and improve efficiency.  

---

## Setup & Run
```bash
git clone https://github.com/<username>/CIE_Prototype.git
cd CIE_Prototype
python -m venv venv          # optional but recommended
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
Tech Stack

Python 3.13+, Streamlit, FAISS, Sentence Transformers, NumPy, JSON
