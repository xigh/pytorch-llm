# PyTorch Inference Engine (Gemma3 270M)

A project based on the book **[Build a Large Language Model (from Scratch)](https://github.com/rasbt/LLMs-from-scratch)** by Sebastian Raschka.

This repository takes one of the notebooks and turns it into a **standalone Python codebase** written **entirely in PyTorch**, making it easier to read, reuse, and experiment with.

![Python](https://img.shields.io/badge/python-blue)
![PyTorch](https://img.shields.io/badge/pytorch-red)

---

## ✨ Goals

* Provide a **clear and educational implementation** of a minimalist LLM.
* Remove external dependencies (except PyTorch) to keep the code **self-contained** and **easy to run**.
* Serve as a foundation for experimenting with new ideas and **deepening the understanding of language models**.

---

## 📂 Project Structure

```
.
├── config.py
├── forward.py
├── gpa.py
├── model3.py
├── modelload.py
├── modelsize.py
├── rmsnorm.py
├── rope.py
├── tokenizer.py
├── transformer.py
├── main.py
├── README.md
└── requirements.txt
````

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/xigh/pytorch-llm.git
cd pytorch-llm
uv venv .venv --seed
source .venv/bin/activate
pip install -r requirements.txt
````

---

## 🚀 Usage

```bash
usage: main.py [-h] [--repo-id REPO_ID] [--prompt PROMPT] [--show-debug]
        [--show-logits SHOW_LOGITS] [--verbose] [--compile]

options:
  -h, --help                    show this help message and exit
  --repo-id REPO_ID             ID du modèle à utiliser
  --prompt PROMPT               prompt
  --show-debug                  show debugging logs
  --show-logits k               show {k} logits
  --verbose                     show timings
  --compile                     compile model
```

### known repo ids :

```bash
"google/gemma-3-270m-it"
"google/gemma-3-1B-it"
```

Example to run an inference request:

```bash
$ python main.py --prompt "How many legs does a duck have?"
A duck has 4 legs.
```

(Yes... tiny model thinks a duck has 4 legs 🦆😂 - it’s just a toy demo!)

---

## 📖 References

* Sebastian Raschka - *Build a Large Language Model (From Scratch)*
* Original repo: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

---

## 🛠️ Next Steps

TODO
