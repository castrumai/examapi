import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import json
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
import re
import random

load_dotenv()

# --- Ortam Değişkenleri ve Konfigürasyon ---
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- YENİ ASİSTAN ID'LERİ ---
OPENAI_ASSISTANT_ID_OPEN_ENDED = "asst_LSwG1illPGhtA5qacwKxfg9n"
OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE = "asst_8TAc0RbpxzRH1hIaT2VCAfi6"
OPENAI_ASSISTANT_ID_ANSWER_CHECKER = "asst_7CJsIplTLNhBfWDYwexRJXSe"

# --- Kontroller ---
if not SUPABASE_URL: raise ValueError("SUPABASE_URL ortam değişkeni ayarlanmamış.")
if not SUPABASE_KEY: raise ValueError("SUPABASE_KEY (SUPABASE_ANON_KEY) ortam değişkeni ayarlanmamış.")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış.")

# --- İstemci Başlatma ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def _run_openai_assistant(assistant_id: str, user_message_content: str) -> str:
    try:
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message_content
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
            messages = await client.beta.threads.messages.list(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == "assistant":
                    for content in msg.content:
                        if content.type == 'text':
                            return content.text.value.strip()
            return ""
        else:
            raise ValueError(f"OpenAI Asistanı işlemi tamamlayamadı. Durum: {run.status}")
    except Exception as e:
        print(f"OpenAI asistanı çalıştırılırken hata: {e}")
        raise


async def generate_open_ended_questions_with_openai_assistant(number_of_questions: int, question_topic: str) -> Dict[str, List[str]]:
    assistant_id = OPENAI_ASSISTANT_ID_OPEN_ENDED
    user_message = f"""Aşağıdaki bilgi kaynağını kullanarak, {number_of_questions} adet açık uçlu soru üret ve her bir soru için ideal bir cevap oluştur. Cevapların kısa ve öz olmalı. Yanıtını sadece şu JSON formatında ver:

{{
  "questions": ["Soru 1", "Soru 2", ...],
  "correct_answers": ["Cevap 1", "Cevap 2", ...]
}}

Bilgi kaynağı sınav sistemine önceden tanımlı, bu nedenle tekrar bilgi istemene gerek yok.

Konu: {question_topic}
"""

    response_text = await _run_openai_assistant(assistant_id, user_message)

    try:
        # Eğer OpenAI cevabı ```json ... ``` formatındaysa json kısmını ayıkla
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = response_text

        parsed_response = json.loads(json_string)

        if isinstance(parsed_response, dict) and "questions" in parsed_response and "correct_answers" in parsed_response:
            return parsed_response
        else:
            raise ValueError("Açık uçlu soru asistanından beklenen formatta yanıt alınamadı.")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Açık uçlu soru yanıtı işlenirken hata: {e} - Gelen yanıt: {response_text}")
        raise HTTPException(status_code=500, detail="Açık uçlu soru asistanından geçerli bir JSON yanıtı alınamadı.")



async def generate_multiple_choice_questions_with_openai_assistant(number_of_questions: int, number_of_choices: int, question_topic: str) -> Dict[str, Any]:
    assistant_id = OPENAI_ASSISTANT_ID_MULTIPLE_CHOICE
    # user_message güncellendi, JSON formatındaki anahtarlar "questions" ve "options" olarak değiştirildi
    user_message = f"""GÖREV: Sınav verisi hazırla. Konu: {question_topic}.
KURAL 1: Tam olarak {number_of_questions} soru üret.
KURAL 2: Her soru için tam olarak {number_of_choices} şık üret.
KURAL 3 (KRİTİK): Her sorunun doğru cevabı, şıklar listesinin HER ZAMAN İLK elemanı olmalıdır.
KURAL 4 (KRİTİK): Şıklar harf (A, B) içermeyen, sadece metinden oluşan string'ler olmalıdır.
ÇIKTI DİLİ: Üretilen tüm sorular ve şıklar tamamen İngilizce olmalıdır.
Yanıtını yalnızca şu JSON formatında ver: {{"questions":["..."],"options":[["Correct Option", "Distractor 1"]]}}
""" # <-- Buradaki JSON örneği güncellendi

    response_text = await _run_openai_assistant(assistant_id, user_message)

    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = response_text
            
        parsed_response = json.loads(json_string)

        # Buradaki kontrol ve atamalar "questions" ve "options" olarak değiştirildi
        if "questions" not in parsed_response or "options" not in parsed_response:
             raise ValueError("Asistandan beklenen formatta yanıt alınamadı (İngilizce anahtarlar 'questions' ve 'options' bekleniyordu).")

        questions = parsed_response["questions"]
        unshuffled_choices_list = parsed_response["options"] # Değişken adı da daha açıklayıcı yapıldı

        final_choices = []
        final_correct_answers = []

        for choices_list_for_question in unshuffled_choices_list: # Döngü değişkeni adı da güncellendi
            if not choices_list_for_question:
                continue

            correct_answer_text = choices_list_for_question[0]
            random.shuffle(choices_list_for_question) # Şıkları karıştır

            # Doğru cevabın karıştırma sonrası yeni indeksini bul
            correct_answer_index = choices_list_for_question.index(correct_answer_text)
            correct_answer_letter = chr(ord('A') + correct_answer_index)
            final_correct_answers.append(correct_answer_letter)

            # Şıkları harflerle formatla (A), B), ...)
            lettered_choices = [f"{chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices_list_for_question)]
            final_choices.append(lettered_choices)

        return {
            "questions": questions,
            "choices": final_choices, # main.py'nin beklediği anahtar "choices"
            "correct_answers": final_correct_answers
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Çoktan seçmeli soru yanıtı işlenirken hata: {e} - Gelen yanıt: {response_text}")
        raise HTTPException(status_code=500, detail=f"Çoktan seçmeli soru asistanından geçerli bir JSON yanıtı alınamadı: {e}")


async def check_answers_with_openai_assistant(questions: List[str], correct_answers: List[str], answers: List[str]) -> List[str]:
    assistant_id = OPENAI_ASSISTANT_ID_ANSWER_CHECKER
    
    input_data = {
        "questions": questions,
        "correct_answers": correct_answers,
        "answers": answers
    }
    user_message = json.dumps(input_data, ensure_ascii=False)

    response_text = await _run_openai_assistant(assistant_id, user_message)

    try:
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = response_text

        parsed_response = json.loads(json_string)
        
        if isinstance(parsed_response, list) and all(isinstance(item, str) for item in parsed_response):
            return parsed_response
        else:
            raise ValueError("Cevap kontrol asistanından beklenen formatta yanıt alınamadı (string listesi değil).")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Cevap kontrol yanıtı işlenirken hata: {e} - Gelen yanıt: {response_text}")
        raise HTTPException(status_code=500, detail="Cevap kontrol asistanından geçerli bir JSON yanıtı alınamadı.")


# --- Veritabanı İşlemleri (Geri Eklendi) ---

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

async def append_question_to_record(student_name: str, question_type: str, new_question: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    questions = record.get('questions', []) if record else []
    questions.append(new_question)
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "questions": questions
    })

async def append_answer_to_record(student_name: str, question_type: str, new_answer: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    answers = record.get('answers', []) if record else []
    answers.append(new_answer)
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "answers": answers
    })

async def update_question_in_record(student_name: str, question_type: str, question_index: int, value: str, correct_answer: Optional[str] = None) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        return None

    # Soruyu güncelle
    questions = record.get('questions', [])
    if not isinstance(questions, list):
        questions = []
    while len(questions) <= question_index:
        questions.append(None)
    questions[question_index] = value
    record['questions'] = questions

    # Eğer yeni bir doğru cevap verilmişse, onu da güncelle
    if correct_answer is not None:
        correct_answers = record.get('correct_answers', [])
        if not isinstance(correct_answers, list):
            correct_answers = []
        while len(correct_answers) <= question_index:
            correct_answers.append(None)
        correct_answers[question_index] = correct_answer
        record['correct_answers'] = correct_answers

    return await upsert_exam_record(record)

async def update_all_questions_in_record(student_name: str, question_type: str, new_questions: List[str], new_correct_answers: Optional[List[str]] = None) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}

    record['questions'] = new_questions
    
    # Eğer yeni doğru cevaplar listesi verilmişse, onu da güncelle
    if new_correct_answers is not None:
        if len(new_questions) != len(new_correct_answers):
            raise ValueError("Soru sayısı ile doğru cevap sayısı eşleşmelidir.")
        record['correct_answers'] = new_correct_answers
    
    return await upsert_exam_record(record)


async def update_choice_in_record(student_name: str, question_type: str, question_index: int, choice_index: int, value: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record or not isinstance(record.get('choices'), list):
        return None

    choices = record['choices']
    if not (isinstance(choices, list) and question_index < len(choices) and isinstance(choices[question_index], list) and choice_index < len(choices[question_index])):
        return None
    
    # Şık metnini harfiyle birlikte günceller
    prefix = f"{chr(ord('A') + choice_index)}) "
    choices[question_index][choice_index] = f"{prefix}{value}"
    record['choices'] = choices

    return await upsert_exam_record(record)

async def update_choices_for_single_question_in_record(student_name: str, question_type: str, question_index: int, new_choices: List[str]) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record or not isinstance(record.get('choices'), list):
        return None

    choices = record.get('choices', [])
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
        record['answers'] = answers

    while len(answers) <= index:
        answers.append(None)

    answers[index] = answer
    
    record['answers'] = answers
    
    return await upsert_exam_record(record)

async def update_answers_bulk(student_name: str, question_type: str, new_answers: List[str]) -> dict | None:
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "answers": new_answers
    })

async def update_result(student_name: str, question_type: str, index: int, result: str) -> dict | None:
    record = await get_student_exam_record(student_name, question_type)
    if not record or not isinstance(record.get('results'), list):
        return None
    results = record['results']
    if index < len(results):
        results[index] = result
        record['results'] = results
        return await upsert_exam_record(record)
    return None

async def update_results_bulk(student_name: str, question_type: str, new_results: List[str]) -> dict | None:
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "results": new_results
    })

async def update_plagiarism_violations_in_record(student_name: str, question_type: str, violation_text: str) -> dict | None:
    return await upsert_exam_record({
        "student_name": student_name,
        "question_type": question_type,
        "plagiarism_violations": violation_text
    })

async def get_question(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and isinstance(record.get('questions'), list) and index < len(record['questions']):
        return record['questions'][index]
    return None

async def get_questions_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    return record.get('questions') if record and isinstance(record.get('questions'), list) else None

async def get_choice(student_name: str, question_type: str, question_index: int, choice_index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and isinstance(record.get('choices'), list) and question_index < len(record['choices']):
        if isinstance(record['choices'][question_index], list) and choice_index < len(record['choices'][question_index]):
            return record['choices'][question_index][choice_index]
    return None

async def get_answer(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and isinstance(record.get('answers'), list) and index < len(record['answers']):
        return record['answers'][index]
    return None

async def get_answers_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    return record.get('answers') if record and isinstance(record.get('answers'), list) else None

async def get_result(student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    if record and isinstance(record.get('results'), list) and index < len(record['results']):
        return record['results'][index]
    return None

async def get_results_all(student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(student_name, question_type)
    return record.get('results') if record and isinstance(record.get('results'), list) else None

async def get_total_score(student_name: str, question_type: str) -> float | None:
    record = await get_student_exam_record(student_name, question_type)
    return record.get('total_score') if record and 'total_score' in record else None

async def get_plagiarism_violations(student_name: str, question_type: str) -> str | None:
    record = await get_student_exam_record(student_name, question_type)
    return record.get('plagiarism_violations') if record and 'plagiarism_violations' in record else None


async def update_correct_answer_in_record(student_name: str, question_type: str, index: int, correct_answer: str) -> dict | None:
    """Belirli bir sorunun doğru cevabını günceller."""
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        return None

    correct_answers = record.get('correct_answers', [])
    if not isinstance(correct_answers, list):
        correct_answers = []
    
    while len(correct_answers) <= index:
        correct_answers.append(None)
        
    correct_answers[index] = correct_answer
    record['correct_answers'] = correct_answers
    
    return await upsert_exam_record(record)

async def update_all_correct_answers_in_record(student_name: str, question_type: str, new_correct_answers: List[str]) -> dict | None:
    """Bir sınavdaki tüm doğru cevapları toplu olarak günceller."""
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        record = {"student_name": student_name, "question_type": question_type}
    
    record['correct_answers'] = new_correct_answers
    return await upsert_exam_record(record)

async def get_correct_answer(student_name: str, question_type: str, index: int) -> str | None:
    """Belirli bir sorunun doğru cevabını döndürür."""
    record = await get_student_exam_record(student_name, question_type)
    if record and isinstance(record.get('correct_answers'), list) and index < len(record['correct_answers']):
        return record['correct_answers'][index]
    return None

async def get_correct_answers_all(student_name: str, question_type: str) -> List[str] | None:
    """Bir sınavdaki tüm doğru cevapları döndürür."""
    record = await get_student_exam_record(student_name, question_type)
    return record.get('correct_answers') if record and isinstance(record.get('correct_answers'), list) else None


async def delete_single_question(student_name: str, question_type: str, index: int) -> dict | None:
    """Belirtilen indeksteki bir soruyu ve ilgili tüm verilerini siler."""
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        return None

    questions = record.get('questions')
    
    # Yeni Kontroller
    if not questions: # Liste boş veya None ise
        raise ValueError("Silinecek soru bulunmuyor (soru listesi boş).")
        
    if not (0 <= index < len(questions)):
        raise ValueError(f"Geçersiz indeks: {index}. Soru sayısı: {len(questions)}.")

    # İlgili tüm listelerden belirtilen indeksteki elemanı sil
    for key in ['questions', 'correct_answers', 'answers', 'results', 'choices']:
        if key in record and isinstance(record.get(key), list) and index < len(record[key]):
            record[key].pop(index)

    return await upsert_exam_record(record)


async def delete_all_questions(student_name: str, question_type: str) -> dict | None:
    """Belirtilen sınav türündeki tüm soruları ve ilgili verileri null olarak ayarlar."""
    record = await get_student_exam_record(student_name, question_type)
    if not record:
        return None

    # Yeni Kontrol
    if not record.get('questions'):
        raise ValueError("Silinecek soru bulunmuyor (soru listesi zaten boş).")

    # İlgili tüm listeleri ve puanı null olarak ayarla
    record['questions'] = None
    record['correct_answers'] = None
    record['answers'] = None
    record['results'] = None
    record['choices'] = None
    record['total_score'] = None

    return await upsert_exam_record(record)
