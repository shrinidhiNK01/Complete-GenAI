
# GenAI Foundations

## What is Artificial Intelligence?
Artificial Intelligence (AI) means making computers behave intelligently.

Examples:
- ChatGPT
- Self-driving cars
- Voice assistants

## What is Machine Learning?
Machine Learning allows systems to learn patterns from data.

Types:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### Supervised Learning
Uses labeled data.

Example:
Input -> Email
Output -> Spam / Not Spam

Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- SVM

## Deep Learning

Deep Learning uses Neural Networks.

Components:
- Input Layer
- Hidden Layer
- Output Layer
- Weights
- Bias
- Activation Functions

Activation Functions:
- Sigmoid
- ReLU
- Tanh
- Softmax

## Generative AI

Generative AI creates NEW content.

Types:
- Text Generation
- Image Generation
- Audio Generation
- Video Generation

Models:
- GPT
- Claude
- Gemini
- Llama

## LLM Pipeline

Raw Data
  ↓
Tokenization
  ↓
Embeddings
  ↓
Transformer Training
  ↓
Fine-tuning
  ↓
Inference

## Important Terms

### Token
Smallest text unit.

### Context Window
Amount of text model can remember.

### Hallucination
Fake confident answers.

### Temperature
Creativity control.


# GenAI Foundations — Complete Beginner to Advanced Explanation

---

# 1. What is Artificial Intelligence (AI)?

Artificial Intelligence (AI) is the field of making machines perform tasks that normally require human intelligence.

Human Intelligence includes:

* Learning
* Reasoning
* Problem Solving
* Understanding Language
* Recognizing Images
* Making Decisions

AI tries to imitate these abilities using algorithms and data.

## Examples of AI

### ChatGPT

Understands questions and generates human-like responses.

### Self-Driving Cars

Can detect roads, traffic lights, pedestrians, and drive automatically.

### Voice Assistants

Examples:

* Siri
* Alexa
* Google Assistant

Can understand speech and perform actions.

---

## AI Hierarchy

```text
Artificial Intelligence (AI)
        │
        ▼
Machine Learning (ML)
        │
        ▼
Deep Learning (DL)
        │
        ▼
Generative AI
        │
        ▼
Large Language Models (LLMs)
```

Think of it like:

```text
Vehicle
 └── Car
      └── Sports Car
           └── Ferrari
```

Similarly:

```text
AI
 └── ML
      └── DL
           └── Generative AI
                └── GPT
```

---

# 2. What is Machine Learning (ML)?

Machine Learning is a subset of AI where systems learn patterns from data instead of being explicitly programmed.

---

## Traditional Programming

```text
Rules + Data
      ↓
  Program
      ↓
 Output
```

Example:

```text
If marks > 40
Pass
Else
Fail
```

Rules are manually written.

---

## Machine Learning

```text
Data + Answers
      ↓
 Training
      ↓
 Model
      ↓
 Prediction
```

The machine learns rules automatically.

---

# Types of Machine Learning

---

# A. Supervised Learning

Uses labeled data.

Labeled means:

```text
Input + Correct Output
```

Example:

```text
Email → Spam
Email → Not Spam
```

The model learns from known answers.

---

## Diagram

```text
Training Data

Email 1 → Spam
Email 2 → Spam
Email 3 → Not Spam
Email 4 → Not Spam

        ↓

Machine Learning Model

        ↓

New Email

        ↓

Spam / Not Spam
```

---

## Applications

* Spam Detection
* House Price Prediction
* Disease Prediction
* Stock Prediction

---

## Algorithms

### Linear Regression

Used for numerical prediction.

Example:

```text
House Size → Price
```

Formula:

genui{"math_block_widget_always_prefetch_v2":{"content":"y=mx+b"}}

Where:

* y = output
* m = slope
* x = input
* b = bias/intercept

---

### Logistic Regression

Used for classification.

Example:

```text
Pass / Fail
Spam / Not Spam
```

Output:

```text
0 or 1
```

---

### Decision Tree

Works like asking questions.

```text
Is age > 18?
     |
    Yes
     |
Has income > 50k?
     |
    Yes
     |
 Approved
```

---

### Random Forest

Many Decision Trees combined together.

```text
Tree 1
Tree 2
Tree 3
Tree 4

   ↓

Majority Vote

   ↓

Prediction
```

More accurate than a single tree.

---

### SVM (Support Vector Machine)

Finds the best boundary separating classes.

```text
Cats  ● ● ●

---------- Best Boundary ----------

Dogs  ▲ ▲ ▲
```

---

# B. Unsupervised Learning

No labels.

Only data is provided.

```text
Customer A
Customer B
Customer C
Customer D
```

Machine finds hidden patterns.

---

## Example

Customer Segmentation

```text
Young Customers
Middle Age Customers
Senior Customers
```

---

## Algorithms

### K-Means Clustering

Groups similar data together.

```text
Data Points

● ● ●

▲ ▲ ▲

■ ■ ■
```

Creates clusters automatically.

---

### PCA

Reduces dimensions.

Example:

```text
100 Features
     ↓
10 Important Features
```

Makes processing faster.

---

# C. Reinforcement Learning

Learning through rewards and punishments.

---

## Example

Training a dog

```text
Correct Action
      ↓
Reward

Wrong Action
      ↓
Punishment
```

---

## RL Components

### Agent

Learner

Example:

```text
Robot
```

### Environment

World where agent acts.

Example:

```text
Road
Game
Factory
```

### Action

What agent does.

```text
Move Left
Move Right
Brake
Accelerate
```

### Reward

Score received.

```text
+10
-5
```

---

## Diagram

```text
Agent
  │
 Action
  │
  ▼

Environment
  │
Reward
  │
  ▼

Agent Learns
```

---

## Applications

* Self Driving Cars
* Chess
* Robotics
* Game AI

---

# 3. Deep Learning

Deep Learning is a subset of Machine Learning that uses Neural Networks.

Inspired by the human brain.

---

## Biological Neuron

```text
Dendrites
    ↓
 Neuron
    ↓
 Output
```

---

## Artificial Neuron

```text
Inputs
  ↓
Weights
  ↓
Sum
  ↓
Activation Function
  ↓
Output
```

---

# Neural Network Structure

```text
Input Layer
      │
      ▼

Hidden Layer 1
      │
      ▼

Hidden Layer 2
      │
      ▼

Output Layer
```

---

## Example

Cat Detection

```text
Image
  ↓
Input Layer
  ↓
Hidden Layers
  ↓
Cat / Not Cat
```

---

# Components of Neural Networks

---

## Input Layer

Receives data.

Example:

```text
Age
Income
Gender
```

---

## Hidden Layers

Perform computations.

More layers = deeper network.

---

## Output Layer

Final prediction.

Example:

```text
Spam
Not Spam
```

---

## Weights

Importance of each input.

```text
Input × Weight
```

Higher weight = more influence.

---

## Bias

Extra adjustment value.

Helps improve predictions.

Formula:

```text
Output = Weight × Input + Bias
```

---

# Activation Functions

Activation decides whether a neuron should fire.

---

## Sigmoid

Output between:

```text
0 and 1
```

Shape:

```text
S Curve
```

Useful for probabilities.

---

## ReLU

Formula:

```text
f(x)=max(0,x)
```

f(x)=\max(0,x)

Most popular activation function.

---

## Tanh

Range:

```text
-1 to +1
```

Centered around zero.

---

## Softmax

Converts outputs into probabilities.

Example:

```text
Cat = 0.8
Dog = 0.1
Bird = 0.1
```

Total:

```text
1.0
```

---

# Deep Learning Architectures

---

## CNN (Convolutional Neural Networks)

Used for Images.

Applications:

* Face Recognition
* Medical Imaging
* Object Detection

Diagram:

```text
Image
 ↓
Convolution
 ↓
Pooling
 ↓
Dense Layer
 ↓
Prediction
```

---

## RNN (Recurrent Neural Networks)

Used for sequences.

Applications:

* Language Processing
* Translation
* Speech Recognition

Diagram:

```text
Word1 → Word2 → Word3
```

Remembers previous information.

---

## LSTM

Advanced RNN.

Can remember long-term information.

Used in:

* Translation
* Time Series Forecasting

---

# 4. Generative AI

Traditional AI predicts.

Generative AI creates.

---

## Traditional AI

```text
Input Image
      ↓
Cat or Dog
```

Output:

Classification

---

## Generative AI

```text
Prompt
      ↓
AI Model
      ↓
New Content
```

Creates something new.

---

# Types of Generative AI

---

## Text Generation

Prompt:

```text
Write a story
```

Output:

New story generated.

---

## Image Generation

Prompt:

```text
A cat in space
```

Output:

New image generated.

---

## Audio Generation

Prompt:

```text
Create piano music
```

Output:

New audio.

---

## Video Generation

Prompt:

```text
Create a movie trailer
```

Output:

New video.

---

# Popular Generative AI Models

* GPT
* Claude
* Gemini
* Llama

---

# 5. Large Language Models (LLMs)

LLMs are Deep Learning models trained on huge amounts of text.

Goal:

```text
Predict Next Token
```

Example:

```text
I love to eat
```

Next token:

```text
pizza
rice
apple
```

Model predicts the most likely next word/token.

---

# LLM Pipeline (Complete)

```text
Raw Data
   ↓
Cleaning
   ↓
Tokenization
   ↓
Embeddings
   ↓
Transformer
   ↓
Pretraining
   ↓
Fine Tuning
   ↓
Inference
```

---

# Step 1: Raw Data

Collected from:

* Books
* Websites
* Research Papers
* Documentation
* Articles

Example:

```text
10+ TB of text
```

---

# Step 2: Data Cleaning

Removes:

* Duplicates
* Spam
* Bad data
* Corrupt text

---

# Step 3: Tokenization

Converts text into tokens.

Example:

```text
I love AI
```

Becomes:

```text
[I]
[love]
[AI]
```

Sometimes:

```text
unbelievable

↓

un
believ
able
```

---

# Step 4: Embeddings

Computers understand numbers, not words.

Words become vectors.

Example:

```text
Cat

↓

[0.23, 1.45, -0.67 ...]
```

---

## Embedding Space

```text
King ------ Queen

Man ------ Woman
```

Similar words are placed close together.

---

# Step 5: Transformer Architecture

The biggest breakthrough in AI.

Introduced in 2017.

Paper:

### Attention Is All You Need

---

## Transformer Diagram

```text
Input Tokens
      ↓
Embeddings
      ↓
Self Attention
      ↓
Feed Forward
      ↓
Output
```

---

# Self-Attention

Allows words to look at other words.

Sentence:

```text
The animal didn't cross the road because it was tired.
```

"It" refers to:

```text
animal
```

Attention helps understand this relationship.

---

## QKV Mechanism

Each token creates:

```text
Query (Q)
Key (K)
Value (V)
```

Then:

```text
Q × K
   ↓
Attention Score
   ↓
Weighted V
```

Produces contextual understanding.

---

# Multi-Head Attention

Many attention mechanisms run in parallel.

```text
Head 1
Head 2
Head 3
Head 4
```

Each learns different relationships.

---

# Step 6: Pretraining

Model learns:

```text
Predict Missing Tokens
```

Example:

```text
The sky is _____
```

Answer:

```text
blue
```

Repeated billions of times.

---

# Step 7: Fine-Tuning

Specialized training.

Examples:

```text
Medical Chatbot
Coding Assistant
Legal Assistant
```

---

# Step 8: Inference

Real-time usage.

```text
User Prompt
      ↓
Model
      ↓
Response
```

This stage happens when you use ChatGPT.

---

# 6. Important Terms

---

## Token

Smallest unit processed by model.

Examples:

```text
Hello
```

≈ 1 token

```text
Artificial Intelligence
```

≈ multiple tokens

---

## Context Window

Amount of information model can remember in one conversation.

Example:

```text
32K Tokens
128K Tokens
1M Tokens
```

Larger context = remembers more text.

---

## Hallucination

When AI generates incorrect information confidently.

Example:

```text
AI invents a fake book
AI creates fake facts
```

Reason:

AI predicts text, not truth.

---

## Temperature

Controls randomness.

### Low Temperature

```text
0.1
```

* More accurate
* Less creative

---

### Medium Temperature

```text
0.5
```

Balanced

---

### High Temperature

```text
1.0+
```

* More creative
* More random

---

# Modern GenAI Ecosystem

```text
Prompt
   ↓
Tokenizer
   ↓
Embedding Model
   ↓
Transformer LLM
   ↓
Output Tokens
   ↓
Response
```

Enhanced with:

```text
Prompt Engineering
RAG
Function Calling
Agents
Memory
Tools
Multimodal AI
```

---

# Advanced Concepts

## RAG (Retrieval-Augmented Generation)

```text
Question
    ↓
Search Database
    ↓
Relevant Documents
    ↓
LLM
    ↓
Answer
```

Reduces hallucinations.

---

## Agents

AI systems that can:

* Think
* Plan
* Use tools
* Execute tasks

Example:

```text
User Request
     ↓
Agent Planning
     ↓
Search
     ↓
Analysis
     ↓
Final Answer
```

---

## Multimodal AI

Can understand multiple data types.

```text
Text
Image
Audio
Video
```

Input:

```text
Image + Question
```

Output:

```text
Explanation
```

---

# One-Line Summary of Everything

```text
AI = Making machines intelligent

ML = Learning from data

DL = Neural networks with many layers

Generative AI = Creating new content

LLM = Predicting next tokens using transformers

Transformer = Core architecture behind modern AI

RAG = LLM + External Knowledge

Agents = LLM + Planning + Tools

Multimodal AI = Text + Images + Audio + Video
```

This hierarchy is the complete roadmap from basic AI concepts to the advanced technologies powering modern systems such as ChatGPT, Gemini, Claude, and other large language models.
