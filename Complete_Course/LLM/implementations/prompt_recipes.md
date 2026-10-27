Prompt Engineering Recipes (Simple English)
=========================================

1) Classification (single label)
- Prompt: "Classify the sentiment of the following review as Positive or Negative:\n\nREVIEW: {text}\nAnswer:" 
- Few-shot: add two examples above the review.

2) Summarization (short)
- Prompt: "Summarize the text below in one short paragraph:\n\n{text}\n\nSummary:" 

3) Extractive QA
- System: "You are a helpful assistant that finds exact answers in the text. Return the short answer or 'NOT FOUND'."
- Prompt: "Context: {document}\nQuestion: {question}\nAnswer:" 

4) Step-by-step reasoning (chain-of-thought)
- Include: "Think step-by-step:" before the question to encourage intermediate reasoning. Use sparingly; prefer fine-tuning for reliable reasoning.

5) Safety guard
- After generation, run a filter step to detect hate, sexual, or violent content. If detected, return a safe fallback message.

Tips:
- Keep examples short and consistent.
- Use explicit instructions (format, length, tone).
- Test different temperatures and sampling settings.
