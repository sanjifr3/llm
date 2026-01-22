Two python modules

## ingest.py
- Read in knowledge base
- Turn documents into chunks
- Vectorize the chunks
- Store in chroma

## answer.py
- Two key functions:
   - fetch_context(question)
   - answer_question(question, history)

# Also a gradio app called app.py

## Two commands:
1. From the implementation directory, `uv run ingest.py`
2. From the week5 directory, `uv run app.py`