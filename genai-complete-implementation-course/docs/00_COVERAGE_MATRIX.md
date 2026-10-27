# 00. Coverage Matrix

This file maps the supplied roadmap into this course. The goal is to make sure no major area is missing.

## Foundation Videos

Covered in:

- `01_BEGINNER_TO_ADVANCED_GENAI_DEEP_NOTES.md`
- `02_RAG_END_TO_END_DEEP_IMPLEMENTATION.md`

Roadmap areas:

- Generative AI roadmap.
- AI vs ML vs DL vs GenAI.
- Classical AI vs modern AI.
- Language modeling history.
- Seq2Seq.
- RNN.
- LSTM.
- GRU.
- Transformer.
- LLM training.
- Diffusion models.
- Text-to-video.

## API and Provider Videos

Covered in:

- `01_BEGINNER_TO_ADVANCED_GENAI_DEEP_NOTES.md`
- `08_MULTIMODAL_VOICE_VIDEO_IMPLEMENTATION.md`
- `projects/advanced_rag_llmops/app/rag/generator.py`

Roadmap areas:

- OpenAI API style integration.
- Gemini text generation.
- Gemini image-to-text.
- Gemini embeddings.
- Safety settings.
- SORA-style text-to-video concepts.

The implementation project uses a local extractive generator by default so it can run without an API key. The optional OpenAI generator is isolated behind `OpenAIGenerator`.

## RAG Videos

Covered in:

- `02_RAG_END_TO_END_DEEP_IMPLEMENTATION.md`
- `03_CODE_WALKTHROUGH_LINE_BY_LINE.md`
- `04_ADVANCED_RAG_IMPLEMENTATION_RECIPES.md`
- `projects/advanced_rag_llmops/`

Roadmap areas:

- End-to-end RAG architecture.
- Ingestion.
- Chunking.
- Embeddings.
- Retrieval.
- Generation.
- FAISS-style local index.
- Open-source local RAG.
- LangChain RAG.
- LlamaIndex RAG.
- Haystack RAG.
- Pinecone, Weaviate, Qdrant, LanceDB, AstraDB, MongoDB vector search.
- Real-time RAG sync.
- Multi-document chat.
- E-commerce chatbot.
- Graph RAG with Neo4j.

## Advanced RAG Videos

Covered in:

- `04_ADVANCED_RAG_IMPLEMENTATION_RECIPES.md`
- `projects/advanced_rag_llmops/app/rag/retrievers.py`
- `projects/advanced_rag_llmops/app/rag/bm25.py`
- `projects/advanced_rag_llmops/app/rag/vector_store.py`

Roadmap areas:

- Hybrid search.
- Ensemble retrieval.
- BM25.
- Reranking.
- Cross encoders.
- Cohere-style reranking.
- Sentence transformers reranking.
- Flash reranking.
- Merger retriever.
- LongContextReorder.
- Lost-in-the-middle problem.
- RAG fusion.
- Contextual compression retriever.
- Self-querying retriever.
- Parent document retriever.
- Sentence window retriever.
- HyDE.
- Corrective RAG.
- Self-RAG.
- Agentic RAG.

## LangChain, LlamaIndex, Haystack

Covered in:

- `04_ADVANCED_RAG_IMPLEMENTATION_RECIPES.md`
- `05_AGENTS_LANGGRAPH_MULTI_AGENT_IMPLEMENTATION.md`

This project implements the ideas from scratch first. After that, LangChain/LlamaIndex/Haystack become easier because you know what every abstraction is doing.

## LangGraph and Agents

Covered in:

- `05_AGENTS_LANGGRAPH_MULTI_AGENT_IMPLEMENTATION.md`
- `projects/advanced_rag_llmops/app/agents/simple_graph.py`
- `projects/advanced_rag_llmops/app/agents/tools.py`

Roadmap areas:

- Agents.
- Tools.
- ReAct.
- Self-ask.
- Custom tools.
- LangGraph from scratch.
- RAG in LangGraph.
- Memory.
- Structured output.
- Finance bot.
- Human-in-the-loop.
- Checkpoints.
- Corrective RAG.
- Agentic RAG.
- Self-RAG.
- SQL agent.
- Multi-agent network.
- Collaborative agents.
- Supervisor multi-agent system.
- Doctor appointment booking agent.
- Research agent.

## Fine-Tuning and Alignment

Covered in:

- `06_FINETUNING_ALIGNMENT_IMPLEMENTATION_GUIDE.md`

Roadmap areas:

- Pretraining.
- Transfer learning.
- Fine-tuning vs RAG vs agents.
- Hugging Face.
- BERT fine-tuning.
- Distillation.
- Quantization.
- LoRA.
- QLoRA.
- LLaMA Factory.
- Unsloth.
- Axolotl.
- OpenAI fine-tuning.
- Gemini/Vertex AI fine-tuning.
- Small language model fine-tuning.
- Multimodal fine-tuning.
- Embedding fine-tuning.
- RLHF.
- RLAIF.
- DPO.
- ORPO.
- PPO.

## LLMOps and Deployment

Covered in:

- `07_LLMOPS_DEPLOYMENT_MULTIPLE_CLOUDS.md`
- `projects/advanced_rag_llmops/Dockerfile`
- `projects/advanced_rag_llmops/docker-compose.yml`
- `projects/advanced_rag_llmops/.github/workflows/ci.yml`

Roadmap areas:

- Modular RAG.
- Config.
- Logger.
- FastAPI.
- Swagger testing.
- UI integration.
- Pytest.
- Evaluation strategy.
- Guardrails.
- AWS ECS/Fargate.
- AWS EKS.
- Azure Jenkins deployment.
- Jenkins with Docker.
- GitHub Actions.
- CI/CD.

## Multimodal, Voice, Video, Dialogflow, Coding Agents

Covered in:

- `08_MULTIMODAL_VOICE_VIDEO_IMPLEMENTATION.md`

Roadmap areas:

- Multimodal RAG.
- Video processing RAG.
- Extract image/table/text from documents.
- Multimodal summarizer.
- MultiVectorRetriever.
- Gemini vision style workflows.
- Multilingual voice assistant.
- Dialogflow weather chatbot.
- Customer support email agent.
- AI coding-agent workflow.

