# 🧠 AI-Powered Resume Screening & Improvement System

A production-quality Streamlit web application that screens resumes against
a job description, ranks candidates, computes ATS scores, identifies skill
gaps, and generates personalized, explainable suggestions to improve a
resume — including AI-rewritten sections.

---

## ✨ Features

- 📄 Upload one or many PDF resumes + a job description (PDF or TXT)
- 🔍 Resume parsing: name, email, phone, LinkedIn, GitHub, skills, education,
  projects, experience, certifications, achievements
- 🧹 Text cleaning & preprocessing (stopword removal, lemmatization, lowercasing)
- 🤖 Semantic matching using `sentence-transformers` (`all-MiniLM-L6-v2`) + cosine similarity
- 📊 ATS score (0–100) from formatting, keywords, skills, education, projects, experience, certifications
- 🧩 Skill gap analysis: existing / missing / recommended skills
- 💡 Resume improvement suggestions, each with a "why" explanation
- ✍️ AI-assisted rewriting of Summary, Projects, Skills, and Experience (facts preserved)
- 🏆 Multi-resume ranking (Match Score, ATS Score, Final Score)
- 📈 Interactive Plotly dashboard
- 🧾 Downloadable PDF report per candidate
- 🛡️ Robust error handling (empty/scanned PDFs, missing sections, oversized files)

---

## 🏗️ Tech Stack

| Purpose                  | Library                          |
|---------------------------|-----------------------------------|
| Web UI                   | Streamlit                         |
| Data handling             | Pandas, NumPy                     |
| PDF parsing               | pdfplumber, PyMuPDF                |
| NLP / preprocessing       | spaCy                             |
| Semantic embeddings       | Sentence-Transformers (MiniLM-L6) |
| Similarity / ML utilities | scikit-learn                       |
| Charts                    | Plotly                            |
| DOCX export               | python-docx                       |
| PDF report generation     | reportlab                          |

---

## 📁 Folder Structure

```
resume_screening_ai/
│
├── app.py                 # Streamlit entry point (built in Phase 10)
├── requirements.txt        # Python dependencies
├── config.py               # Central paths, weights, keyword lists
│
├── models/                 # Cached / loaded ML models (spaCy, SentenceTransformer)
├── utils/                  # Reusable helper functions (file I/O, validation, cleaning)
├── modules/                # Core business logic (parser, scorer, matcher, report gen)
│
├── uploads/                 # User-uploaded resumes & job descriptions
├── outputs/                 # Intermediate JSON / parsed data
├── reports/                 # Generated PDF reports
├── assets/                  # Logos, icons, static images
│
└── README.md
```

---

## ⚙️ Installation

1. **Clone / copy the project**, then move into the folder:
   ```bash
   cd resume_screening_ai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy English model** (required for parsing/NLP):
   ```bash
   python -m spacy download en_core_web_sm
   ```

> The first time you run the app, `sentence-transformers` will download the
> `all-MiniLM-L6-v2` model (~80MB) automatically and cache it locally.

---

## ▶️ Running the Project

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

---

## 🧪 Development Roadmap (build order)

This project is being built incrementally, phase by phase:

- [x] **Phase 1** — Project structure, `requirements.txt`, `README.md`
- [x] **Phase 2** — Resume parser
- [x] **Phase 3** — Text preprocessing
- [x] **Phase 4** — Semantic similarity
- [x] **Phase 5** — ATS scoring engine
- [x] **Phase 6** — Skill gap analysis
- [x] **Phase 7** — Resume improvement engine
- [x] **Phase 8** — Dashboard
- [ ] **Phase 9** — PDF report generation
- [ ] **Phase 10** — Final integration (`app.py`)

---

## 🖼️ Screenshots

> _Placeholders — screenshots will be added once the UI (Phase 8/10) is built._

| Upload Page | Dashboard | Candidate Ranking |
|-------------|-----------|--------------------|
| _(screenshot)_ | _(screenshot)_ | _(screenshot)_ |

---

## 🚀 Future Enhancements

- 🤖 AI chatbot to answer questions about a candidate's resume
- ❓ Interview question generator based on the JD + resume
- ✉️ Cover letter generator
- 🎯 Job recommendation system based on resume content
- 🔄 Resume version comparison (before vs. after improvement)

---

## 📄 License

This project is provided as-is for educational and portfolio purposes.
