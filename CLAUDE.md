# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**FlyLLM** is a comprehensive knowledge base repository focused on **LLM (Large Language Models), Agent systems, and algorithmic problem-solving**. The repository serves as both a technical documentation hub and interview preparation resource.

### Core Content Areas

- **LLM Research & Technical Documentation** - In-depth technical articles on transformers, attention mechanisms, fine-tuning techniques, RLHF, and inference optimization
- **Agent System Design** - Research on agent architectures, tool calling mechanisms, function calling, skill systems, and memory mechanisms
- **Algorithm Practice** - LeetCode problem solutions with detailed explanations and complexity analysis
- **Documentation Subprojects** - Several full-featured ML/NLP projects including transformers library documentation

## Key Directories

### LLM Technical Documentation

- **`/llm/`** - Comprehensive LLM research documents (legacy organization, ~100+ topics)
  - Covers: Tokenizers, Transformer architecture, Attention mechanisms, Training methods, Inference optimization, RAG systems, Agent frameworks, RLHF, Memory systems
  - Each file is a self-contained technical deep-dive on a specific topic

- **`/llm_v1/`** - New structured LLM documentation with sequential learning path
  - `001_tokenizer.md` - Tokenization fundamentals
  - `002_self-attention-mechanism.md` - Core attention mathematics
  - `003_mha-gqa-comparison.md` - Attention architecture comparisons
  - `code/` - Python implementations for key algorithms (MHA, GQA, etc.)
  - **Note**: This is the actively maintained version with ongoing expansion

### Algorithm Practice

- **`/leetcode/`** - LeetCode problem solutions and explanations
  - `hot100.md` - Complete categorized list of LeetCode Hot 100 problems
  - Individual problem files using the format: `{problem-number}.{problem-name}.md`
  - Each solution includes problem description, approach, complexity analysis, and implementation
  - Organized by difficulty and topic for systematic preparation

### Documentation Subprojects

- **`/docs/`** - Full-featured documentation and implementation projects
  - **`transformers/`** - Complete HuggingFace transformers library documentation (full project)
  - **`nanochat/`** - Lightweight LLM chatbot implementation
  - **`autoresearch/`** - Automated research and documentation generation tools
  - **`Engram/`** - Memory-related research and implementations

### Supporting Directories

- **`/image/`** - Images, diagrams, and visual assets used across documentation
- **`/interview/`** - Interview preparation materials (currently being organized)

## Common Commands

### Quick Navigation & Search

```bash
# Find files by topic (example: searching for attention-related content)
find . -name "*.md" -type f | xargs grep -l "attention" | head -10

# Search within llm directory
grep -r "LoRA\|lora" --include="*.md" llm/

# Search within llm_v1
grep -r "self-attention" --include="*.md" llm_v1/

# List recently modified files
ls -lt $(find . -name "*.md" -type f) | head -10

# Find LeetCode problems by topic
grep -r "dynamic programming" --include="*.md" leetcode/
```

### LLM Documentation Management

```bash
# Find all tokenizer-related documents
find llm -name "*token*" -type f

# Find RLHF training documents
find llm -name "*RLHF*" -o -name "*DPO*" -o -name "*SFT*"

# Check for duplicate topics between llm and llm_v1
find llm -name "*.md" -exec basename {} \; | sort | uniq -d
```

### Subprojects

#### Transformers Documentation (`/docs/transformers/`)

Full HuggingFace transformers library documentation project:

```bash
cd docs/transformers

# Run tests
make test

# Check code quality
make quality

# Fix style issues
make style
```

#### NanoChat (`/docs/nanochat/`)

Lightweight LLM chatbot implementation:

```bash
cd docs/nanochat

# Check project-specific documentation in the directory
# for installation and usage instructions
```

#### AutoResearch (`/docs/autoresearch/`)

Automated research and documentation generation tools:

```bash
cd docs/autoresearch

# Check README for tool usage and setup instructions
```

#### Engram (`/docs/Engram/`)

Memory-related research and implementations:

```bash
cd docs/Engram

# Check project documentation for memory system details
```

### LeetCode

Solutions and explanations in `/leetcode/`:

```bash
# View LeetCode Hot 100 categorized list
cat leetcode/hot100.md

# Search for problems by topic (e.g., DP, tree, graph)
grep -r "dynamic programming" --include="*.md" leetcode/ | head -10

# View solution for a specific problem
cat leetcode/1.两数之和.md

# List all problem files
ls leetcode/*.md
```

Each problem file includes:
- Problem description
- Solution approach and intuition
- Time/space complexity analysis
- Code implementation with explanations
- Common pitfalls and edge cases

## Architecture and Design Patterns

### LLM Documentation Structure

The `/llm/` and `/llm_v1/` directories cover:

- **Tokenization** - BPE, SentencePiece, WordPiece, and subword algorithms
- **Transformer Architecture** - Attention mechanisms (MHA, GQA, MQA), positional encodings, and optimization
- **Training** - Pretraining, SFT (Supervised Fine-Tuning), LoRA/QLoRA, RLHF, DPO
- **Inference** - KV cache optimization, quantization, continuous batching, vLLM/SGLang
- **RAG Systems** - Retrieval design, vector databases, reranking, query expansion
- **Agent Frameworks** - ReAct architecture, tool calling, skill systems, memory mechanisms
- **Memory Systems** - Short-term/long-term memory, retrieval strategies, evaluation metrics

### Document Creation Guidelines

When creating new documentation:

#### LLM Documentation (`/llm_v1/`)
- Use sequential numbering: `001_{topic}.md`, `002_{topic}.md`, etc.
- Include code examples in accompanying `code/` directory
- Each document should be self-contained with clear progression

#### LeetCode Solutions
- Use format: `{problem-number}.{problem-name}.md` (e.g., `1.两数之和.md`)
- Follow existing template structure in `leetcode/0.模板.md`
- Include complexity analysis and multiple approaches if applicable

#### Legacy LLM Docs (`/llm/`)
- Use descriptive hyphenated names: `{topic}.md` (e.g., `BPE-algorithm.md`)
- Focus on specific, well-defined topics
- Include code examples and mathematical formulations

## Code Style and Standards

- **Documentation** - Technical documents primarily use Chinese explanations
- **Code examples** - Python implementations with clear comments and complexity analysis
- **Math formulas** - LaTeX syntax for mathematical expressions
- **File naming** - Use appropriate patterns per directory (see above)
- **Content organization** - Clear structure with sections for introduction, technical details, and examples

## Quick Reference

### Finding Content by Topic

| Topic | Directory | Key Files |
|-------|-----------|-----------|
| Tokenization | `/llm/`, `/llm_v1/` | `BPE-algorithm.md`, `001_tokenizer.md` |
| Transformer Architecture | `/llm/`, `/llm_v1/` | `TransformerArchitecture.md`, `002_self-attention-mechanism.md` |
| Attention Mechanisms | `/llm/`, `/llm_v1/code/` | `MHA.md`, `GQA.md`, `MultiHeadAttention.md` |
| Training & Fine-tuning | `/llm/` | `LoRA-Principle.md`, `QLoRA-Principle.md`, `SFT_VS_RLHF.md` |
| RLHF & Alignment | `/llm/` | `RLHF-FullProcess.md`, `Reward-Model-Training.md` |
| RAG & Memory | `/llm/` | `RAG.md`, `MemGPTLayeredMemoryAndVirtualContext.md` |
| Agent Systems | `/llm/` | `ReActFramework.md`, `ToolCalling.md` |
| LeetCode Solutions | `/leetcode/` | `hot100.md`, various problem files |
| Subprojects | `/docs/` | `transformers/`, `nanochat/`, `autoresearch/`, `Engram/` |

### Common Development Tasks

```bash
# Start new LLM_v1 document
# - Create numbered file in llm_v1/
# - Add code examples in llm_v1/code/ if needed

# Add new LeetCode solution
# - Follow existing naming convention
# - Use template from leetcode/0.模板.md

# Search for implementation references
find . -name "*.py" -type f | xargs grep -l "self-attention" | head -5

# Check git status
git status
```

## Important Notes

- **Active Development** - Focus on `/llm_v1/` for new LLM content (structured learning path)
- **Knowledge Repository** - This is primarily documentation, not an executable software project (except subprojects)
- **Chinese Content** - Main documentation is in Chinese, targeting Chinese-speaking audience
- **Multiple Versions** - Some content exists in both `/llm/` and `/llm_v1/` - prefer `/llm_v1/` for learning
- **Subproject Independence** - Each subproject in `/docs/` has its own dependencies and build system

## Last Updated

- **Last Revision**: 2026-03-19
- **Current Focus**: Expanding `/llm_v1/` with sequential numbered documentation
- **Active Areas**: LLM training techniques, agent systems, algorithm practice
