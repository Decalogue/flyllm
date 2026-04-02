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

- **`/llm/`** - Structured LLM documentation with sequential learning path
  - `001_tokenizer.md` - Tokenization fundamentals
  - `002_self-attention-mechanism.md` - Core attention mathematics
  - `003_mha-gqa-comparison.md` - Attention architecture comparisons
  - `004_rope-alibi-position-encoding.md` - Position encoding methods
  - And more numbered documents covering inference, fine-tuning, and optimization
  - Each file is a self-contained technical deep-dive on a specific topic
  - Covers: Tokenizers, Transformer architecture, Attention mechanisms, Training methods, Inference optimization, RAG systems, Agent frameworks, RLHF, Memory systems

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

# List numbered documents in sequential order
ls -lh llm/0*.md | awk '{print $9, "-", $5}'
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

The `/llm/` directory covers:

- **Tokenization** - BPE, SentencePiece, WordPiece, and subword algorithms
- **Transformer Architecture** - Attention mechanisms (MHA, GQA, MQA), positional encodings, and optimization
- **Training** - Pretraining, SFT (Supervised Fine-Tuning), LoRA/QLoRA, RLHF, DPO
- **Inference** - KV cache optimization, quantization, continuous batching, vLLM/SGLang
- **RAG Systems** - Retrieval design, vector databases, reranking, query expansion
- **Agent Frameworks** - ReAct architecture, tool calling, skill systems, memory mechanisms
- **Memory Systems** - Short-term/long-term memory, retrieval strategies, evaluation metrics

### Document Creation Guidelines

When creating new documentation:

#### LLM Documentation (`/llm/`)
- Use sequential numbering: `001_{topic}.md`, `002_{topic}.md`, etc.
- Follow existing format in `/llm/` directory
- Each document should be self-contained with clear progression

#### LeetCode Solutions
- Use format: `{problem-number}.{problem-name}.md` (e.g., `1.两数之和.md`)
- Follow existing template structure in `leetcode/0.模板.md`
- Include complexity analysis and multiple approaches if applicable

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
| Tokenization | `/llm/` | `001_tokenizer.md` |
| Transformer Architecture | `/llm/` | `002_self-attention-mechanism.md`, `003_mha-gqa-comparison.md` |
| Attention Mechanisms | `/llm/` | `MHA.md`, `GQA.md`, `MultiHeadAttention.md` |
| Training & Fine-tuning | `/llm/` | `014_finetuning-full-vs-efficient.md`, `015_lora-principle-tuning.md` |
| RLHF & Alignment | `/llm/` | `RLHF-FullProcess.md`, `Reward-Model-Training.md` |
| RAG & Memory | `/llm/` | `RAG.md`, `MemGPTLayeredMemoryAndVirtualContext.md` |
| Agent Systems | `/llm/` | `ReActFramework.md`, `ToolCalling.md` |
| LeetCode Solutions | `/leetcode/` | `hot100.md`, various problem files |
| Subprojects | `/docs/` | `transformers/`, `nanochat/`, `autoresearch/`, `Engram/` |

### Common Development Tasks

```bash
# Start new LLM document
# - Follow sequential numbering: 001_{topic}.md, 002_{topic}.md, etc.
# - Add to /llm/ directory

# Add new LeetCode solution
# - Follow existing naming convention
# - Use template from leetcode/0.模板.md

# Search for implementation references
find . -name "*.py" -type f | xargs grep -l "self-attention" | head -5

# Check git status
git status
```

### LeetCode 5-Day Sprint - State Recovery

```bash
# Check current sprint progress
cat workspace/leetcode/coach/session-state.json | python3 -m json.tool

# View completed Q&A records
ls workspace/leetcode/coach/*.md

# Resume sprint from last session
# 1. Read session state
# 2. Check current problem in 5-day plan
grep -A 5 '"current_problem"' workspace/leetcode/coach/session-state.json
cat leetcode/hot100-sprint-plan.md | grep -A 10 "current_problem"

# Review today's problems
day=$(grep '"active_day"' workspace/leetcode/coach/session-state.json | cut -d: -f2 | tr -d ',' | xargs)
echo "Day $day Problems:"
grep -A 50 "### Day $day:" leetcode/hot100-sprint-plan.md | grep -E '^\| [0-9]+' | head -20
```

## Important Notes

- **Active Development** - `/llm/` directory contains structured documentation with sequential numbering
- **Knowledge Repository** - This is primarily documentation, not an executable software project (except subprojects)
- **Chinese Content** - Main documentation is in Chinese, targeting Chinese-speaking audience
- **Structured Organization** - LLM docs use sequential numbering (001, 002, etc.) for clear learning path
- **Subproject Independence** - Each subproject in `/docs/` has its own dependencies and build system

## Last Updated

- **Last Revision**: 2026-04-02
- **Current Focus**: Expanding `/llm/` with sequentially numbered documentation for structured learning path
- **Active Areas**: LLM training techniques (LoRA, QLoRA), inference optimization, agent systems, algorithm practice
- **Knowledge Graph**: Unified knowledge tracking system located at `.claude/skills/interview-master/memory/Rain/knowledge_graph.json`
