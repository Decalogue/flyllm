# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a knowledge base repository focused on **LLM (Large Language Models), Agent systems, and Memory mechanisms**. The repository contains:

- **LLM research documents** - Technical deep-dives on transformers, fine-tuning, RLHF, inference optimization, and RAG systems
- **Agent framework research** - Documentation on agent architectures, tool calling, function calling, skill systems, and memory mechanisms
- **Interview preparation** - Common LLM/ML interview questions with detailed answers
- **LeetCode solutions** - Algorithm problems and solutions
- **Cursor skills** - Specialized tools for knowledge card generation, LeetCode coaching, and self-improving agents
- **Documentation subprojects** - Includes a full transformers library documentation (docs/transformers) and other ML projects

## Key Directories

- `/llm/` - Core LLM research and technical documentation
  - `transformer/` - Transformer architecture, attention mechanisms, positional encoding
  - `training/` - Pretraining, fine-tuning (SFT, LoRA, QLoRA), RLHF, DPO
  - `inference/` - KV cache, quantization, vLLM/SGLang, continuous batching
  - `rag/` - Retrieval design, vector databases, reranking, query expansion
  - `agent/` - ReAct, tool calling, multi-agent systems, memory mechanisms
- `/agent/` - Agent system research and design documents
- `/leetcode/` - LeetCode problem solutions and explanations
  - Organized by problem number (e.g., `0001-two-sum.md`)
  - Each includes problem description, approach, complexity analysis, and code
- `/docs/` - Documentation subprojects
  - `transformers/` - HuggingFace transformers library documentation
  - `nanochat/` - Lightweight chatbot implementation
  - `autoresearch/` - Automated research tools
- `/solution/` - Structured solution templates for common ML/LLM interview questions
  - `transformer/` - Transformer architecture questions
  - `agent/` - Agent system design questions
  - `distributed/` - Distributed training questions
  - `rlhf/` - RLHF and alignment questions
  - `memory/` - Memory mechanisms and RAG questions

## Common Commands

### Quick Navigation & Search

```bash
# Find files by topic (example: searching for LoRA-related content)
find . -name "*.md" -type f | xargs grep -l "LoRA\|lora" | head -10

# Search for specific interview questions
grep -r "attention.*mechanism" --include="*.md" llm/

# List recently modified files
ls -lt $(find . -name "*.md" -type f) | head -10
```

### Transformers Subproject

Located in `/docs/transformers/` - this is a full HuggingFace transformers documentation project:

```bash
cd docs/transformers

# Install dependencies
pip install -e .

# Run code quality checks
make quality

# Auto-fix code style issues
make style

# Run tests
make test

# Run specific test file
python -m pytest -n auto --dist=loadfile -s -v ./tests/test_specific.py

# Build documentation
make docs
```

### LeetCode

Solutions and explanations are in `/leetcode/` organized by problem number:

```bash
# View solution for a specific problem
cat leetcode/0001-two-sum.md

# Search for problems by topic (e.g., DP, tree, graph)
grep -r "dynamic programming" --include="*.md" leetcode/ | head -10
```

Each problem file includes:
- Problem description
- Solution approach and intuition
- Time/space complexity analysis
- Code implementation with explanations
- Common pitfalls and edge cases

### Document Creation Guidelines

When creating new documentation:

```bash
# Use consistent file naming
# LLM topics: lowercase with hyphens (e.g., "rope-positional-encoding.md")
# LeetCode: 4-digit-number + problem-name (e.g., "0001-two-sum.md")
# Interview solutions: topic + number + keyword (e.g., "q01_MHA_to_GQA.md")

## Architecture and Design Patterns

### LLM Documentation Structure

The `/llm/` directory follows a topic-based organization:

- **Core concepts** - Transformer architecture, attention mechanisms, positional encoding
- **Training** - Pretraining, fine-tuning (SFT, LoRA, QLoRA), RLHF, DPO
- **Inference** - KV cache, quantization, vLLM/SGLang, continuous batching
- **RAG systems** - Retrieval design, vector databases, reranking, query expansion
- **Agent frameworks** - ReAct, tool calling, function calling, skill systems, memory mechanisms

Each document typically includes:
- Problem statement or concept explanation
- Technical details and mathematical formulations
- Implementation considerations
- Code examples where applicable

### Agent System Design

Agent research focuses on modular architectures:

- **Core modules** - Planning, tool use, memory, reflection
- **Function calling** - Schema design, multi-turn handling, error recovery, parallel execution
- **Skill systems** - Skill extraction, skill libraries, skill selection and composition
- **Memory mechanisms** - Short-term vs long-term memory, retrieval strategies

- **Multi-agent coordination** - Task allocation, communication protocols, autonomy levels

### Interview Question Standards

Solution templates in `/solution/` follow structured format:

1. **Question format**: `q<number>_<topic>_<keyword>.md`
   - Example: `q01_MHA_to_GQA_evolution.md`

2. **Content structure**:
   - Clear problem statement and context
   - Mathematical derivations where applicable
   - Step-by-step explanations
   - Code examples in Python/PyTorch
   - Performance considerations and trade-offs

3. **Common topics organized by category**:
   - `transformer/` - Attention mechanisms, positional encoding, architecture variants
   - `agent/` - ReAct, tool calling, state machines, error handling
   - `distributed/` - ZeRO, tensor/model parallelism, communication patterns
   - `rlhf/` - PPO, reward modeling, alignment techniques
   - `memory/` - RAG, retrieval strategies, memory hierarchies

## Code Style and Standards

- **Documentation** - All technical documents use markdown with Chinese explanations
- **Code examples** - Python implementations with clear comments and complexity analysis
- **Math formulas** - LaTeX rendering via MathJax in HTML outputs
- **Diagrams** - Mermaid or custom CSS for architectural visualizations
- **File naming** - Use lowercase with hyphens for regular docs, numbered prefix for LeetCode and interview questions
- **Content organization** - Each major topic gets its own section with subtopics clearly separated

## Quick Reference

### Finding Content by Topic

| Topic | Directory | Key Files |
|-------|-----------|-----------|
| Transformer Architecture | `/llm/transformer/` | `mha-to-gqa-evolution.md`, `rope-positional-encoding.md` |
| Training & Fine-tuning | `/llm/training/` | `lora-qlora-comparison.md`, `rlhf-three-stage.md` |
| Inference Optimization | `/llm/inference/` | `kv-cache-optimization.md`, `vllm-sglang-comparison.md` |
| RAG & Memory | `/llm/rag/`, `/solution/memory/` | `hybrid-retrieval-rrf.md`, `memgpt-hierarchical-memory.md` |
| Agent Systems | `/agent/`, `/solution/agent/` | `react-state-machine.md`, `tool-calling-parser.md` |
| Distributed Training | `/solution/distributed/` | `zero3-parameter-sharding.md`, `tensor-parallelism.md` |
| LeetCode Solutions | `/leetcode/` | `0001-two-sum.md` to `latest-problem.md` |

### Content Creation Workflow

When adding new content:

1. **Identify the right directory** based on topic (see table above)
2. **Follow naming conventions**:
   - General documentation: lowercase-hyphens.md (e.g., `attention-is-all-you-need.md`)
   - Interview questions: q[00]_[topic]_[keyword].md (e.g., `q07_lora_backward.md`)
   - LeetCode: [4-digit-number]-[problem-name].md (e.g., `0155-min-stack.md`)
3. **Include standard sections**:
   - Problem/concept introduction
   - Technical details and math (if applicable)
   - Implementation examples
   - Complexity analysis
   - References and further reading
4. **Test markdown rendering** for formulas and diagrams before committing

## Important Notes

- This is primarily a **knowledge repository**, not an executable software project
- Root directory has no build system or dependencies
- Each subproject (`docs/transformers/`, `docs/nanochat/`, etc.) has its own dependencies and build commands
- Content is mainly in Chinese, targeting Chinese-speaking audience for interview prep and technical research
- When contributing, focus on clarity and depth of technical explanations
- For formulas and equations, use LaTeX syntax within markdown for proper rendering
