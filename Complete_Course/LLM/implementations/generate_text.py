"""
Generate text with Hugging Face `transformers` pipeline.

Usage:
  python generate_text.py --model gpt2 --prompt "Write a short poem"

This script shows simple decoding controls: temperature, top_k, top_p, max_length.
"""
import argparse
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()

    gen = pipeline('text-generation', model=args.model)
    out = gen(args.prompt, max_length=args.max_length, do_sample=True,
              temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(out[0]['generated_text'])


if __name__ == '__main__':
    main()
