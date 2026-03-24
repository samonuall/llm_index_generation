**Also create project-level README:**

```bash
cat > README.md << 'EOF'
# LLM Index Generation

Evaluating LLM-driven document preprocessing agents for retrieval.

## Agents

### 1. **Baseline** (`baseline`)
- Simple passthrough: 1 chunk per document
- No preprocessing
- Performance floor

### 2. **AI Assistant Baseline** (`ai_assistant_baseline`)
- One-shot LLM code generation
- Tests if GPT-4/Claude can design good chunking in single prompt
- See: `src/agents/ai_assistant_baseline/README.md`

### 3. **Analysis Code Agent** (`analysis_code_agent`)
- Iterative improvement with code execution
- Main experimental agent

## Quick Start

```bash
# Setup
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Add API key to .env
echo "LITELLM_API_KEY=your_key" >> .env

# Run baseline
python src/evaluation/scripts/test_preprocessing_split.py \
    --agent baseline \
    --split paper_retrieval_5000docs

# Run AI assistant
python src/evaluation/scripts/test_preprocessing_split.py \
    --agent ai_assistant_baseline \
    --split paper_retrieval_5000docs

# Run iterative agent
python main.py --agent analysis_code_agent --split paper_retrieval_5000docs --loops 3