# ExamAI API

FastAPI service that generates, evaluates, and manages exam content with OpenAI models, Supabase persistence, and PDF‑backed retrieval. Designed for human instructors to launch and grade assessments quickly while keeping full auditability.

## Highlights
- **AI question generation**: Open-ended, multiple-choice, and verbal questions grounded in course PDFs via Supabase `match_chunks` retrieval.
- **Rubric-first scoring**: Deterministic rubric compiler, automated auditing, and reasoning traces for each answer.
- **Voice to text**: Accepts audio uploads, transcribes with Whisper, and stores responses for verbal exams.
- **Persistent records**: Supabase `exam_records` table stores questions, choices, answers, results, rubrics, and plagiarism flags per student/exam/type.
- **Secure by default**: All routes require `castrumai-apikey` header; environment-driven OpenAI and Supabase keys.

## Architecture
- **API**: `main.py` (FastAPI) with lifespan hook precomputing file-name embeddings for faster semantic lookups.
- **Domain logic**: `examai.py` handles RAG chunk retrieval, rubric compilation, OpenAI calls (`gpt-4.1-mini`, `gpt-4.1-nano`, `text-embedding-3-small`, `whisper-1`), and Supabase operations.
- **Context sources**: Module/file metadata in `MODULE_FILES` and `MODULE_TOPICS`; RAG via Supabase RPC `match_chunks`.
- **Deployment**: `Procfile` runs `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Prerequisites
- Python 3.11+ recommended
- Supabase project with:
  - `exam_records` table (fields like `exam_name`, `student_name`, `question_type`, `questions`, `choices`, `correct_answers`, `answers`, `results`, `evaluation_rubrics`, `plagiarism_violations`, `total_score`)
  - `match_chunks` RPC returning chunk `content`, `file_name`, `module_id`, and vector similarity metadata
- OpenAI account with access to Chat, Embeddings, and Whisper APIs

## Environment
Set the following variables (e.g., in `.env`):

```
OPENAI_API_KEY=...
OPENAI_ASSISTANT_ID_ANSWER_CHECKER=...   # assistant for auditing answers
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
CASTRUMAI_API_KEY=...                    # value clients must send in header castrumai-apikey
PDF_BASE_PATH=./pdfs                     # optional, defaults to ./pdfs
```

## Setup
1) Create and activate a virtualenv  
`python -m venv .venv && source .venv/bin/activate`

2) Install dependencies  
`pip install -r requirements.txt`

3) Run locally  
`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

4) Open docs  
`http://localhost:8000/docs` (Swagger UI) or `/redoc`

## API Quickstart
All requests must include `castrumai-apikey: <CASTRUMAI_API_KEY>`.

### Generate open-ended questions + rubrics
```
curl -X POST http://localhost:8000/generate/open-ended \
  -H "Content-Type: application/json" \
  -H "castrumai-apikey: $CASTRUMAI_API_KEY" \
  -d '{
    "exam_name": "Marine Safety 101",
    "student_name": "Jane Doe",
    "number_of_questions": 3,
    "question_topic": "M1"
  }'
```

### Evaluate answers against rubrics
```
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "castrumai-apikey: $CASTRUMAI_API_KEY" \
  -d '{
    "exam_name": "Marine Safety 101",
    "student_name": "Jane Doe",
    "question_topic": "M1"
  }'
```

### Generate multiple-choice questions
```
curl -X POST http://localhost:8000/generate/mcq \
  -H "Content-Type: application/json" \
  -H "castrumai-apikey: $CASTRUMAI_API_KEY" \
  -d '{
    "exam_name": "Marine Safety 101",
    "student_name": "Jane Doe",
    "number_of_questions": 5,
    "number_of_choices": 4,
    "question_topic": "M2"
  }'
```

### Verbal flow (voice upload + feedback)
- `POST /generate/verbal` to create verbal questions and feedback guides.  
- `POST /answers/voice` multipart upload (mp3/wav/m4a/mp4/webm) to transcribe and store a verbal answer.  
- `POST /feedback/verbal` to generate instructor-facing feedback for recorded answers.

### Record management
- `POST /exam-record` upsert a full record (questions, choices, answers, results, rubrics, scores).  
- `DELETE /exam-record` remove a record.  
- `PUT /update/...` endpoints cover granular updates for questions, choices, answers, correct answers, results, and plagiarism notes.  
- `GET /record`, `/questions`, `/answers`, `/results`, `/score`, `/plagiarism-violations` fetch stored data.

## Data conventions
- `question_type` values the API expects: `"Open Ended"`, `"Multiple Choice"`, `"Verbal Question"`.
- Correct answers for MCQ are stored as letters (A/B/C/…) after shuffling choices.
- Verbal feedback guides are stored in `correct_answers`; student transcriptions are in `answers`.

## Running with Procfile
```
PORT=8000 uvicorn main:app --host 0.0.0.0 --port $PORT
```
Use this for platforms like Heroku/Render that read `Procfile`.

## Tips
- Ensure `initialize_file_name_embeddings()` runs at startup (handled by FastAPI lifespan) so `_find_relevant_files_by_keyword` works without extra OpenAI calls per request.
- Keep Supabase credentials scoped to a service or anon key that has the correct RPC/table permissions.
- For deterministic auditing, keep `OPENAI_ASSISTANT_ID_ANSWER_CHECKER` consistent across environments.

## Testing
`pytest` is available in dependencies, but the project currently ships without test cases. Add endpoint or service tests before production launches.
