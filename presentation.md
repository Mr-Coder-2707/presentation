---
theme: default
style: |
  :root {
    --primary: #ff5caa;
    --secondary: #cbbbe0;
    --accent: #1a003d;
    --background: #fcfcfd;
  }
  section {
    background: linear-gradient(135deg, #fcfcfd 0%, #f8f9fa 100%);
    background-image: 
      radial-gradient(circle at 20% 50%, rgba(255, 92, 170, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 80%, rgba(203, 187, 224, 0.04) 0%, transparent 50%);
    color: #252b39;
  }
  h1, h2 {
    background: linear-gradient(135deg, #ff5caa 0%, #ff8dbb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  strong {
    color: #ff5caa;
  }
  code {
    background: rgba(255, 92, 170, 0.1);
    color: #1a003d;
    padding: 2px 6px;
    border-radius: 4px;
  }
---

<!-- Page 1: Title -->
# 🔍 Information Retrieval System
## Advanced Information Retrieval System

<div style="text-align: center; margin-top: 40px;">

### Flask-Based IR Web Application
**Built with Modern Architecture & Multiple Ranking Algorithms**

<div style="margin-top: 30px; font-size: 0.9em; color: #6b7280;">
Powered by Python • Flask • Bootstrap • Advanced IR Algorithms
</div>

<div style="margin-top: 50px; font-size: 0.95em;">

**Instructor:** Dr. Sarah Hassan

**Project Team:**
| Student Name | Student ID | Serial No. | Hall |
|--------------|------------|------------|------|
| Mahmoud Sabry Mahmoud Ali El-Khawas | 2303006 | 1 | Hall 215 |
| Ibrahim Adel Yahya | 2303142 | 13 | Hall 211 |
| Mohamed El-Gharib Ahmed Hassan Saqr | 2303068 | 6 | Hall 213 |
| Abdulrahman Zakaria Ahmed Mohamed | 2303029 | 21 | Hall 215 |
| Zakaria Khaled Zakaria Senousy | 2303046 | 34 | Hall 215 |

</div>

</div>

---

<!-- Page 2: System Overview -->
## 📋 System Overview

### **What is the System?**
A comprehensive web-based information retrieval system providing an interactive interface for document search using advanced algorithms

### **Core Components:**
- 🌐 **Interactive Web Interface** - Flask + Bootstrap + Modern UI
- 🧠 **Advanced Search Engine** - Multiple algorithms (Boolean, VSM, BM25)
- 📊 **Comprehensive Evaluation System** - Precision, Recall, F1, NDCG, MAP
- 📁 **Multi-format File Processing** - TXT, PDF, DOCX, CSV, JSON
- 🔧 **Professional Text Processing** - Tokenization, Stemming, Stopwords

---

<!-- Page 3: Core Features -->
## ⚡ Core Features

### **1️⃣ Document Management**
- ✅ Upload multiple file types (TXT, PDF, DOCX, CSV, JSON)
- ✅ Load sample data for testing
- ✅ Add documents manually
- ✅ Delete and clear documents

### **2️⃣ Index Building (Inverted Index)**
- ✅ Build professional inverted index
- ✅ Advanced text processing
- ✅ Comprehensive index statistics

---

<!-- Page 4: Search Algorithms -->
## 🔬 Available Search Algorithms

### **1. Boolean Model** 🎯
- Supports logical operations: `AND`, `OR`, `NOT`
- Precise and fast search
- Example: `machine AND learning NOT deep`

### **2. Vector Space Model (VSM)** 📐
- **TF**: Term Frequency
- **TF-IDF**: Term Frequency - Inverse Document Frequency
- **Log TF-IDF**: Logarithmic scaling
- Similarity measurement using Cosine Similarity

### **3. BM25 Algorithm** 🏆
- Advanced ranking algorithm
- Dynamic document length calibration
- Highest accuracy in results

---

<!-- Page 5: Text Processing Pipeline -->
## 🔄 Text Processing Pipeline

### **Processing Stages:**

**1️⃣ Tokenization** - Split text into words
```python
"Machine Learning" → ["machine", "learning"]
```

**2️⃣ Lowercase** - Convert to lowercase
```python
"Learning" → "learning"
```

**3️⃣ Remove Stopwords** - Remove common words
```python
["the", "machine", "is"] → ["machine"]
```

**4️⃣ Stemming** - Reduce words to their root
```python
"running" → "run"
"studies" → "studi"
```

---

<!-- Page 6: Inverted Index Structure -->
## 📚 Inverted Index Structure

### **How Does the Index Work?**

**Example:**
```
Doc 1: "Machine learning is amazing"
Doc 2: "Learning Python programming"
Doc 3: "Machine vision systems"
```

**Inverted Index:**
```python
{
  "machine": [Doc1(pos:0), Doc3(pos:0)],
  "learning": [Doc1(pos:1), Doc2(pos:0)],
  "python": [Doc2(pos:1)],
  "programming": [Doc2(pos:2)],
  "vision": [Doc3(pos:1)],
  "systems": [Doc3(pos:2)]
}
```

---

<!-- Page 7: Search Process -->
## 🔍 Search Process

### **Search Steps:**

**1️⃣ Query Input**
```
User Query: "machine learning algorithms"
```

**2️⃣ Query Processing**
```
Processed: ["machine", "learn", "algorithm"]
```

**3️⃣ Document Retrieval**
```
Retrieve documents containing terms
```

**4️⃣ Score Calculation (Ranking)**
```
TF-IDF, BM25, Boolean matching
```

**5️⃣ Results Ranking**
```
Return top-K ranked documents
```

---

<!-- Page 8: Evaluation Metrics -->
## 📊 Evaluation Metrics

### **Metrics Used:**

**1. Precision** 📏
```
Precision = Relevant Retrieved / Total Retrieved
```

**2. Recall** 🎯
```
Recall = Relevant Retrieved / Total Relevant
```

**3. F1-Score** ⚖️
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**4. NDCG** (Normalized Discounted Cumulative Gain) 🏅

**5. MAP** (Mean Average Precision) 📈

**6. MRR** (Mean Reciprocal Rank) 🔢

---

<!-- Page 9: Web Interface - Main Features -->
## 🌐 Main Interface

### **Interface Sections:**

**1️⃣ System Status Panel** 📊
- Number of loaded documents
- Index status
- Active algorithm
- Index statistics

**2️⃣ Document Management** 📁
- Upload files
- Add texts
- Load sample data
- Clear data

**3️⃣ Search Interface** 🔍
- Smart search bar
- Algorithm selection
- Display ranked results

---

<!-- Page 10: Web Interface - Advanced Features -->
## 🎨 Advanced Interface

### **Interactive Features:**

**🌙 Dark Mode**
- Toggle between dark and light mode
- Animated stars background in dark mode
- Modern and attractive design

**📱 Responsive Design**
- Compatible with all screens
- Responsive design for phones and tablets

**⚡ Real-time Updates**
- Instant status updates
- Live results display
- Interactive notifications

**🎯 Smart Results Display**
- Ranking by relevance
- Score display
- Highlight relevant texts

---

<!-- Page 11: Algorithm Comparison -->
## ⚖️ Algorithm Comparison

### **Comparison Table:**

| Algorithm | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| **Boolean** | ⚡⚡⚡ Very Fast | ⭐⭐ Medium | Exact search |
| **TF** | ⚡⚡ Fast | ⭐⭐⭐ Good | Basic search |
| **TF-IDF** | ⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | Advanced search |
| **Log TF-IDF** | ⚡⚡ Fast | ⭐⭐⭐⭐ Excellent | Balanced search |
| **BM25** | ⚡ Medium | ⭐⭐⭐⭐⭐ Outstanding | Best accuracy |

**💡 Recommendation:** Use **BM25** for maximum accuracy, **TF-IDF** for balance

---

<!-- Page 12: REST API -->
## 🔌 System REST API

### **Available Endpoints:**

**📍 GET `/api/status`**
- Get system status

**📍 POST `/api/load-sample`**
- Load sample data

**📍 POST `/api/upload`**
- Upload new files

**📍 POST `/api/add-document`**
- Add document manually

**📍 POST `/api/build-index`**
- Build index

**📍 POST `/api/search`**
- Execute search
```json
{
  "query": "machine learning",
  "algorithm": "bm25",
  "top_k": 10
}
```

---

<!-- Page 13: Technical Architecture -->
## 🏗️ Technical Architecture

### **Software Components:**

**Backend (Flask)** 🐍
```
app.py              → Flask Application
ir_core/
  ├── pipeline.py        → IR Pipeline
  ├── preprocessor.py    → Text Processing
  ├── dataset_loader.py  → Data Management
  ├── evaluation.py      → Metrics
  └── search_algorithms.py → Algorithms
```

**Frontend** 🎨
```
templates/index.html    → UI Template
static/stars.css       → Animations
Bootstrap 5.3          → UI Framework
```

**Data Processing** 📊
```
NLTK              → Stemming
Custom Tokenizer  → Text Processing
```

---

<!-- Page 14: Performance & Statistics -->
## 📈 Performance & Statistics

### **Index Statistics:**

**🔢 Index Statistics**
- **Total Documents**: Total number of documents
- **Unique Terms**: Number of unique terms
- **Total Terms**: Total number of terms
- **Average Doc Length**: Average document length

**⏱️ Performance Metrics**
- **Index Build Time**: Index building time
- **Search Time**: Search time (in milliseconds)
- **Throughput**: Queries per second

**📊 Search Results**
- **Retrieved Documents**: Retrieved documents
- **Relevance Scores**: Relevance scores
- **Precision & Recall**: Precision and recall

---

<!-- Page 15: Conclusion & Future -->
## 🎯 Conclusion & Future Work

### **✨ What We Achieved:**
- ✅ Complete and functional IR system
- ✅ Multiple advanced algorithms
- ✅ Modern and interactive user interface
- ✅ Comprehensive evaluation system
- ✅ API for integration

### **🚀 Future Enhancements:**
- 🔮 Arabic language support
- 🔮 Machine Learning for ranking
- 🔮 Query Expansion
- 🔮 Semantic Search
- 🔮 Multi-language Support

### **📚 References:**
- Flask Documentation
- Information Retrieval: Modern Approach
- NLTK Library

---

<div style="text-align: center; margin-top: 80px;">

# 🙏 Thank You
## Questions?

<div style="margin-top: 40px; color: #ff5caa;">
💻 GitHub • 📧 Contact • 🌐 Demo
</div>

</div>
