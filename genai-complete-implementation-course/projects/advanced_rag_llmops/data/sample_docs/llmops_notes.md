# LLMOps Notes

Production RAG systems should log request id, user or tenant id, prompt version, model name, retrieval chunk ids, latency, token usage, errors, and safety decisions.

Production systems should not log API keys, passwords, or private personal data unless there is a clear legal reason and strong protection.

Evaluation should check answer correctness, faithfulness, retrieval relevance, source accuracy, latency, and cost.

