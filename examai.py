import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Optional, Dict, Any, Union
from openai import AsyncOpenAI
import asyncio
import json
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_ANON_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID_OPEN_ENDED = os.getenv("OPENAI_ASSISTANT_ID_OPEN_ENDED")
OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_GENERATOR = os.getenv("OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_GENERATOR")
OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_CHOICE_GENERATOR = os.getenv("OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_CHOICE_GENERATOR") # <-- Bu değişkenin tanımlandığı satır
OPENAI_ASSISTANT_ID_ANSWER_CHECKER = os.getenv("OPENAI_ASSISTANT_ID_ANSWER_CHECKER")

if not SUPABASE_URL: raise ValueError("SUPABASE_URL ortam değişkeni ayarlanmamış.")
if not SUPABASE_KEY: raise ValueError("SUPABASE_KEY (SUPABASE_ANON_KEY) ortam değişkeni ayarlanmamış.")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış.")
if not OPENAI_ASSISTANT_ID_OPEN_ENDED: raise ValueError("OPENAI_ASSISTANT_ID_OPEN_ENDED ortam değişkeni ayarlanmamış.")
if not OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_GENERATOR: raise ValueError("OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_GENERATOR ortam değişkeni ayarlanmamış.")
if not OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_CHOICE_GENERATOR: raise ValueError("OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_CHOICE_GENERATOR ortam değişkeni ayarlanmamış.") # <-- Hatayı aldığınız satır
if not OPENAI_ASSISTANT_ID_ANSWER_CHECKER: raise ValueError("OPENAI_ASSISTANT_ID_ANSWER_CHECKER ortam değişkeni ayarlanmamış.")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_student_exam_record(student_name: str, question_type: str) -> dict | None:
    try:
        response = await run_in_threadpool(
            lambda: supabase.table('exam_records').select("*")
            .eq("student_name", student_name)
            .eq("question_type", question_type)
            .single()
            .execute()
        )
        return response.data
    except Exception as e:
        if "PGRST" in str(e) and "0 rows" in str(e):
            return None
        print(f"Sınav kaydı alınırken hata oluştu: {e}")
        return None

async def upsert_exam_record(record_data: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        response = await run_in_threadpool(
            lambda: supabase.table('exam_records').upsert(record_data, on_conflict='student_name,question_type').execute()
        )
        return response.data[0]
    except Exception as e:
        print(f"Sınav kaydı eklenirken/güncellenirken hata oluştu: {e}")
        return None

async def generate_open_ended_questions_with_openai_assistant(student_name: str, number_of_questions: int) -> List[str]:
    assistant_id = OPENAI_ASSISTANT_ID_OPEN_ENDED
    thread = await client.beta.threads.create()
    
    user_message = f"{student_name} için {number_of_questions} adet açık uçlu sınav sorusu oluştur."
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    while run.status in ['queued', 'in_progress', 'cancelling']:
        await asyncio.sleep(1)
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    if run.status == 'completed':
        messages = await client.beta.threads.messages.list(
            thread_id=thread.id
        )
        response_content = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == 'text':
                        response_content += content.text.value + "\n"
                break
        
        try:
            parsed_response = json.loads(response_content.strip())
            
            if isinstance(parsed_response, list):
                questions_list = parsed_response
            elif isinstance(parsed_response, dict) and "questions" in parsed_response:
                questions_list = parsed_response["questions"]
            else:
                raise ValueError("OpenAI'den beklenen formatta yanıt alınamadı: Liste veya 'questions' anahtarı bulunamadı.")
            
            return questions_list
        except json.JSONDecodeError as e:
            print(f"JSON ayrıştırma hatası: {e} - Yanıt içeriği: {response_content}")
            raise ValueError("OpenAI'den beklenen formatta yanıt alınamadı.")
    else:
        raise ValueError(f"OpenAI Asistanı soruyu oluşturamadı. Durum: {run.status}")


async def generate_multiple_choice_questions_with_openai_assistant(student_name: str, number_of_questions: int, number_of_choices: int) -> Dict[str, Any]:
    question_generator_assistant_id = OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_GENERATOR
    choice_generator_assistant_id = OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE_QUESION_CHOICE_GENERATOR

    thread = await client.beta.threads.create()
    
    user_message = f"{student_name} için {number_of_questions} adet çoktan seçmeli sınav sorusu oluştur."
    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=question_generator_assistant_id
    )

    while run.status in ['queued', 'in_progress', 'cancelling']:
        await asyncio.sleep(1)
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    if run.status == 'completed':
        messages = await client.beta.threads.messages.list(
            thread_id=thread.id
        )
        questions_response_content = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == 'text':
                        questions_response_content += content.text.value + "\n"
                break
        
        try:
            generated_questions_data = json.loads(questions_response_content.strip())
            questions_list = [q for q in generated_questions_data["questions"]]

            all_choices = []
            for question_text in questions_list:
                choice_thread = await client.beta.threads.create()
                choice_user_message = f"Soru: '{question_text}' için {number_of_choices} adet seçenek oluştur. Doğru cevabı 'is_correct: true' olarak işaretle."
                await client.beta.threads.messages.create(
                    thread_id=choice_thread.id,
                    role="user",
                    content=choice_user_message
                )

                choice_run = await client.beta.threads.runs.create(
                    thread_id=choice_thread.id,
                    assistant_id=choice_generator_assistant_id
                )

                while choice_run.status in ['queued', 'in_progress', 'cancelling']:
                    await asyncio.sleep(1)
                    choice_run = await client.beta.threads.runs.retrieve(
                        thread_id=choice_thread.id,
                        run_id=choice_run.id
                    )
                
                if choice_run.status == 'completed':
                    choice_messages = await client.beta.threads.messages.list(
                        thread_id=choice_thread.id
                    )
                    choice_response_content = ""
                    for msg in choice_messages.data:
                        if msg.role == "assistant":
                            for content in msg.content:
                                if content.type == 'text':
                                    choice_response_content += content.text.value + "\n"
                            break
                    try:
                        generated_choices_data = json.loads(choice_response_content.strip())
                        choices_for_question = [f"{c['text']} (Correct)" if c['is_correct'] else c['text'] for c in generated_choices_data["choices"]]
                        all_choices.append(choices_for_question)
                    except json.JSONDecodeError as e:
                        print(f"JSON ayrıştırma hatası (seçenekler): {e} - Yanıt içeriği: {choice_response_content}")
                        raise ValueError("OpenAI'den seçenekler için beklenen formatta yanıt alınamadı.")
                else:
                    raise ValueError(f"OpenAI Asistanı seçenekleri oluşturamadı. Durum: {choice_run.status}")

            return {"questions": questions_list, "choices": all_choices}

        except json.JSONDecodeError as e:
            print(f"JSON ayrıştırma hatası (sorular): {e} - Yanıt içeriği: {questions_response_content}")
            raise ValueError("OpenAI'den sorular için beklenen formatta yanıt alınamadı.")
    else:
        raise ValueError(f"OpenAI Asistanı soruları oluşturamadı. Durum: {run.status}")


async def check_answers_with_openai_assistant(student_name: str, question_type: str, questions: List[str], answers: List[str], choices: Optional[List[List[str]]] = None) -> List[Dict[str, Any]]:
    assistant_id = OPENAI_ASSISTANT_ID_ANSWER_CHECKER
    thread = await client.beta.threads.create()

    evaluation_prompt = f"Öğrencinin adı: {student_name}\nSınav Tipi: {question_type}\n\n"
    for i, q in enumerate(questions):
        evaluation_prompt += f"Soru {i+1}: {q}\nCevap {i+1}: {answers[i]}\n"
        if question_type == "Multiple Choice" and choices and i < len(choices):
            evaluation_prompt += f"Seçenekler {i+1}: {', '.join(choices[i])}\n"
    evaluation_prompt += "\nHer cevabı değerlendir. Her cevabın doğru olup olmadığını, eğer yanlışsa neden yanlış olduğunu ve doğru cevabı belirt. Her değerlendirmeyi JSON formatında döndür: {\"evaluation_results\": [{\"question_number\": 1, \"is_correct\": true/false, \"feedback\": \"...\", \"correct_answer\": \"...\"}]}"

    await client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=evaluation_prompt
    )

    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    while run.status in ['queued', 'in_progress', 'cancelling']:
        await asyncio.sleep(1)
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    if run.status == 'completed':
        messages = await client.beta.threads.messages.list(
            thread_id=thread.id
        )
        response_content = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for content in msg.content:
                    if content.type == 'text':
                        response_content += content.text.value + "\n"
                break
        
        try:
            evaluation_results = json.loads(response_content.strip())
            return evaluation_results["evaluation_results"]
        except json.JSONDecodeError as e:
            print(f"JSON ayrıştırma hatası (değerlendirme): {e} - Yanıt içeriği: {response_content}")
            raise ValueError("OpenAI'den değerlendirme için beklenen formatta yanıt alınamadı.")
    else:
        raise ValueError(f"OpenAI Asistanı cevapları değerlendiremedi. Durum: {run.status}")


async def append_question_to_record(student_name: str, question_type: str, new_question: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    questions = []
    if record:
        questions_from_record = record.get('questions')
        if questions_from_record is not None:
            questions = questions_from_record
    
    questions.append(new_question)
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "questions": questions
    })

async def append_answer_to_record(student_name: str, question_type: str, new_answer: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    answers = []
    if record:
        answers_from_record = record.get('answers')
        if answers_from_record is not None:
            answers = answers_from_record
    
    answers.append(new_answer)
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "answers": answers
    })

async def update_question_in_record(student_name: str, question_type: str, question_index: int, value: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type, "questions": []}

    questions = record.get('questions')
    if not isinstance(questions, list):
        questions = []

    while len(questions) <= question_index:
        questions.append(None)

    questions[question_index] = value
    record['questions'] = questions
    return await upsert_exam_record(record)

async def update_all_questions_in_record(student_name: str, question_type: str, new_questions: List[str]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['questions'] = new_questions
    return await upsert_exam_record(record)

async def update_choice_in_record(student_name: str, question_type: str, question_index: int, choice_index: int, value: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type, "choices": []}

    choices = record.get('choices')
    if not isinstance(choices, list):
        choices = []

    while len(choices) <= question_index:
        choices.append([])

    question_choices = choices[question_index]
    if not isinstance(question_choices, list):
        question_choices = []

    while len(question_choices) <= choice_index:
        question_choices.append("")

    question_choices[choice_index] = value
    choices[question_index] = question_choices
    record['choices'] = choices
    return await upsert_exam_record(record)

async def update_choices_for_single_question_in_record(student_name: str, question_type: str, question_index: int, new_choices: List[str]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type, "choices": []}

    choices = record.get('choices')
    if not isinstance(choices, list):
        choices = []

    while len(choices) <= question_index:
        choices.append([])

    choices[question_index] = new_choices
    record['choices'] = choices
    return await upsert_exam_record(record)

async def update_all_choices_in_record(student_name: str, question_type: str, all_new_choices: List[List[str]]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['choices'] = all_new_choices
    return await upsert_exam_record(record)

async def update_answer(student_name: str, question_type: str, index: int, answer: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type, "answers": []}

    answers = record.get('answers')
    if not isinstance(answers, list):
        answers = []

    while len(answers) <= index:
        answers.append(None)

    answers[index] = answer
    record['answers'] = answers
    return await upsert_exam_record(record)

async def update_answers_bulk(student_name: str, question_type: str, new_answers: List[str]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['answers'] = new_answers
    return await upsert_exam_record(record)

async def update_result(student_name: str, question_type: str, index: int, result: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type, "results": []}

    results = record.get('results')
    if not isinstance(results, list):
        results = []

    while len(results) <= index:
        results.append(None)

    results[index] = result
    record['results'] = results
    return await upsert_exam_record(record)

async def update_results_bulk(student_name: str, question_type: str, new_results: List[str]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['results'] = new_results
    return await upsert_exam_record(record)

async def update_plagiarism_violations_in_record(student_name: str, question_type: str, violation_text: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['plagiarism_violations'] = violation_text
    return await upsert_exam_record(record)


async def get_question(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'questions' in record and isinstance(record['questions'], list) and len(record['questions']) > index:
        return record['questions'][index]
    return None

async def get_questions_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'questions' in record and isinstance(record['questions'], list):
        return record['questions']
    return None

async def get_choice(student_name: str, question_type: str, question_index: int, choice_index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'choices' in record and isinstance(record['choices'], list) and len(record['choices']) > question_index:
        if isinstance(record['choices'][question_index], list) and len(record['choices'][question_index]) > choice_index:
            return record['choices'][question_index][choice_index]
    return None

async def get_answer(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'answers' in record and isinstance(record['answers'], list) and len(record['answers']) > index:
        return record['answers'][index]
    return None

async def get_answers_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'answers' in record and isinstance(record['answers'], list):
        return record['answers']
    return None

async def get_result(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'results' in record and isinstance(record['results'], list) and len(record['results']) > index:
        return record['results'][index]
    return None

async def get_results_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'results' in record and isinstance(record['results'], list):
        return record['results']
    return None

async def get_total_score(student_name: str, question_type: str) -> float | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'total_score' in record:
        return record['total_score']
    return None

async def get_plagiarism_violations(student_name: str, question_type: str) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and 'plagiarism_violations' in record:
        return record['plagiarism_violations']
    return None
