// UI copy and per-mode configuration, mirrored from the backend's QUERY_PROMPTS.

export const MODE_LABELS = {
  qa: "Factual Q&A",
  summarize: "Summarization",
  compare: "Comparative Analysis",
  path: "Learning Path",
};

export const MODE_TAB_LABELS = {
  qa: "Factual Q&A",
  summarize: "Summarize",
  compare: "Compare",
  path: "Learning Path",
};

export const MODE_PLACEHOLDERS = {
  qa: "Ask anything about the papers in the corpus…",
  summarize: "Enter a paper name or topic to summarize…",
  compare: "Enter two concepts or papers to compare…",
  path: "Enter a topic to generate a learning path…",
};

export const SAMPLE_QUESTIONS = {
  qa: [
    "What are Naive RAG, Advanced RAG, and Modular RAG?",
    "What is Corrective Retrieval Augmented Generation (CRAG)?",
    "How does multi-head attention work in the Transformer?",
    "What is masked language modeling in BERT?",
  ],
  summarize: [
    "Summarize the three RAG paradigms from the RAG survey paper",
    "Summarize the best practices for building a RAG pipeline",
    "Summarize the key contributions of the BERT paper",
    "Summarize how corrective RAG improves retrieval quality",
  ],
  compare: [
    "Compare Naive RAG and Advanced RAG approaches",
    "Compare BERT and the Transformer from Attention Is All You Need",
    "Compare parametric memory and non-parametric memory in RAG",
    "Compare different retrieval methods used in RAG systems",
  ],
  path: [
    "Give me a learning path to understand RAG from basics to advanced",
    "What papers should I read to understand Transformer-based RAG?",
    "What should I study first to understand retrieval-augmented generation?",
    "Learning path for knowledge-intensive NLP tasks using RAG",
  ],
};

export const LEARNING_PATH_LABELS = [
  "Foundation",
  "Core Methods",
  "Advanced Topics",
  "Applications",
  "Further Reading",
  "Beyond",
];

export const MODES = ["qa", "summarize", "compare", "path"];
