# RAG Chatbot for Alcatel-Lucent Enteprise Products **AIOps**

**AIOps** is a lightweight RAG application that acts as a AI chatbot to support Alcatel-Lucent Enterprise Products.

---

## ✨ Requirements

- Python 3.8 
- OpenAI API keys

---


## 🚀 Installation

Step 1: Create config.py file:

```
# config.py
OPENAI_API_KEY = "<YOUR-OPENAI-KEY>"
```

Step 2: Build the Vector database

```
python .\build_kb.py --folders "YOUR-PATH-TO-REPOSITORIES"

```

Step 3: Start FastAPI server

```
uvicorn chatbot:app --reload

```





---

## 📦 Releases

| Version          | Date       | Notes                       |
|------------------|------------|-----------------------------|
| v1.0.0           | 2025-05-5  | Initial release             |


---

## 📄 License

```
Copyright (c) Samuel Yip Kah Yean <2025>

This software is licensed for personal, non-commercial use only.

You are NOT permitted to:
- Use this software for any commercial purposes.
- Modify, adapt, reverse-engineer, or create derivative works.
- Distribute, sublicense, or share this software.

All rights are reserved by the author.

For commercial licensing or permission inquiries, please contact:
kahyean.yip@gmail.com
```


