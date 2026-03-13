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
- `/agent/` - Agent system research and design documents
- `/leetcode/` - LeetCode problem solutions and explanations
- `/docs/` - Subprojects including transformers docs, nanochat, autoresearch, and skills
- `/.cursor/skills/` - Cursor editor skills for automated workflows

## Common Commands

### Knowledge Cards System

Generate and publish knowledge cards to Xiaohongshu:

```bash
# Initialize a new knowledge card project
python .cursor/skills/knowledge-cards/scripts/init_project.py <topic_name>

# Generate HTML cards from content template
python .cursor/skills/knowledge-cards/scripts/card_generator.py <topic_name> --content ~/.easyclaw/workspace/knowledge-cards/<topic_name>/content_template.json

# Capture screenshots of cards (1080x1440 for Xiaohongshu)
python .cursor/skills/knowledge-cards/scripts/card_capture.py <topic_name>

# Publish to Xiaohongshu
python .cursor/skills/knowledge-cards/scripts/publish_xhs.py --title "title" --desc "description" --images img1.png img2.png
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
# Each problem has its own markdown file with:
# - Problem description
# - Solution approach
# - Time/space complexity analysis
# - Code implementation
```

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

### Knowledge Cards Pipeline

The knowledge-cards skill automates content creation:
1. **Content generation** - MathJax for formulas, DAG visualization for concepts
2. **Screenshot capture** - Playwright automation with 1080x1440 resolution
3. **Social publishing** - Xiaohongshu API integration

## Code Style and Standards

- **Documentation** - All technical documents use markdown with Chinese explanations
- **Code examples** - Python implementations with clear comments and complexity analysis
- **Math formulas** - LaTeX rendering via MathJax in HTML outputs
- **Diagrams** - Mermaid or custom CSS for architectural visualizations

## Important Notes

- This is primarily a **knowledge repository**, not an executable software project
- Root directory has no build system or dependencies
- Each subproject (docs/transformers, docs/nanochat, etc.) has its own dependencies and build commands
- Cursor skills provide automation for specific workflows but require proper setup
- Content is mainly in Chinese, targeting Chinese-speaking audience for interview prep and technical research
