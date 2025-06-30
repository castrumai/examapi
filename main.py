import os
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import examai
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from typing import List

load_dotenv()

app = FastAPI(
    title="ExamAI API",
    description="Python tabanlı tüm sınav ve değerlendirme işlemlerini yürüten API.",
    version="1.0.0"
)

CASTRUMAI_API_KEY_HEADER_NAME = "castrumai-apikey"
VALID_CASTRUMAI_API_KEY = os.getenv("CASTRUMAI_API_KEY")

if not VALID_CASTRUMAI_API_KEY:
    raise ValueError("CASTRUMAI_API_KEY ortam değişkeni ayarlanmamış.")

async def verify_castrumai_api_key(castrumai_apikey: Optional[str] = Header(None, alias=CASTRUMAI_API_KEY_HEADER_NAME)):
    if castrumai_apikey != VALID_CASTRUMAI_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz CastrumAI API Anahtarı")
    return True

class OpenEndedQuestionGenerationRequest(BaseModel):
    student_name: str
    number_of_questions: int

class MultipleChoiceQuestionGenerationRequest(BaseModel):
    student_name: str
    number_of_questions: int
    number_of_choices: int

class AnswerEvaluationRequest(BaseModel):
    student_name: str
    question_type: str

class AnswerUpdateRequest(BaseModel):
    student_name: str
    question_type: str
    index: int
    answer: str

class AnswersBulkUpdateRequest(BaseModel):
    student_name: str
    question_type: str
    answers: List[str]

class ResultUpdateRequest(BaseModel):
    student_name: str
    question_type: str
    index: int
    result: str

class ResultsBulkUpdateRequest(BaseModel):
    student_name: str
    question_type: str
    results: List[str]

class CreateExamRecordRequest(BaseModel):
    student_name: str
    question_type: str
    questions: Optional[List[str]] = None
    choices: Optional[List[List[str]]] = None
    answers: Optional[List[str]] = None
    results: Optional[List[str]] = None
    total_score: Optional[float] = 0.0
    plagiarism_violations: Optional[str] = ""

class DeleteExamRecordRequest(BaseModel):
    student_name: str
    question_type: str

class AppendQuestionRequest(BaseModel):
    student_name: str
    question_type: str
    question: str

class AppendAnswerRequest(BaseModel):
    student_name: str
    question_type: str
    answer: str

class UpdateChoiceRequest(BaseModel):
    student_name: str
    question_index: int
    choice_index: int
    value: str

class UpdateQuestionChoicesRequest(BaseModel):
    student_name: str
    question_index: int
    choices: List[str] # Belirli bir sorunun tüm seçenekleri

class UpdateAllChoicesRequest(BaseModel):
    student_name: str
    choices: List[List[str]] # Tüm soruların tüm seçenekleri

class UpdateQuestionValueRequest(BaseModel):
    student_name: str
    question_type: str
    question_index: int
    value: str

class UpdateAllQuestionsRequest(BaseModel):
    student_name: str
    question_type: str
    questions: List[str]

class AddPlagiarismViolationRequest(BaseModel):
    student_name: str
    question_type: str
    violation_text: str

@app.post("/generate/open-ended", summary="Açık uçlu sorular oluşturur ve veri tabanına ekler.")
async def generate_open_ended(
    request: OpenEndedQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        existing_record = await examai.get_student_exam_record(request.student_name, "Open Ended")
        existing_questions = []
        if existing_record and existing_record.get('questions'):
            existing_questions = existing_record['questions']

        newly_generated_questions = await examai.generate_open_ended_questions_with_openai_assistant(
            request.student_name,
            request.number_of_questions
        )

        all_questions = existing_questions + newly_generated_questions
        
        record_data = {
            "student_name": request.student_name,
            "question_type": "Open Ended",
            "questions": all_questions
        }
        
        await examai.upsert_exam_record(record_data)

        return {"message": "Açık uçlu sorular başarıyla üretildi ve mevcut kayda eklendi.", "new_questions_added": newly_generated_questions}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Açık uçlu sorular üretilirken veya kaydedilirken hata oluştu: {e}")


@app.post("/generate/mcq", summary="Çoktan seçmeli sorular oluşturur ve veri tabanına ekler. Bir öğrenci için seçenek sayısı her soru için aynı olmak zorundadır.")
async def generate_mcq(
    request: MultipleChoiceQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        existing_record = await examai.get_student_exam_record(request.student_name, "Multiple Choice")
        existing_questions = []
        existing_choices = []
        if existing_record:
            existing_questions = existing_record.get('questions', [])
            existing_choices = existing_record.get('choices', [])

        newly_generated_mcq_data = await examai.generate_multiple_choice_questions_with_openai_assistant(
            request.student_name,
            request.number_of_questions,
            request.number_of_choices
        )
        
        new_questions = newly_generated_mcq_data.get("questions", [])
        new_choices = newly_generated_mcq_data.get("choices", [])

        if not new_questions:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OpenAI Asistanlarından çoktan seçmeli soru ayrıntıları alınamadı.")

        all_questions = existing_questions + new_questions
        all_choices = existing_choices + new_choices

        record_data = {
            "student_name": request.student_name,
            "question_type": "Multiple Choice",
            "questions": all_questions,
            "choices": all_choices
        }
        
        await examai.upsert_exam_record(record_data)

        return {
            "message": "Çoktan seçmeli sorular başarıyla üretildi ve mevcut kayda eklendi.", 
            "new_questions_added": new_questions, 
            "new_choices_added": new_choices
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Çoktan seçmeli sorular üretilirken veya kaydedilirken hata oluştu: {e}")

@app.post("/evaluate", summary="Verilen cevapları değerlendirir.")
async def evaluate_answers_with_openai(
    request: AnswerEvaluationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        exam_record = await examai.get_student_exam_record(request.student_name, request.question_type)

        if not exam_record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Değerlendirme için sınav kaydı bulunamadı.")

        questions_from_db = exam_record.get('questions', [])
        answers_from_db = exam_record.get('answers', [])
        choices_from_db = None

        if request.question_type == "Multiple Choice":
            choices_from_db = exam_record.get('choices', [])
            if not choices_from_db:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Çoktan seçmeli sorular için seçenekler bulunamadı.")


        if not questions_from_db or not answers_from_db or len(questions_from_db) != len(answers_from_db):
            # Corrected typo here: HTTP_400_BAD_BAD_REQUEST -> HTTP_400_BAD_REQUEST
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Değerlendirmek için eksik veya eşleşmeyen soru/cevaplar.")

        evaluation_results_list = await examai.check_answers_with_openai_assistant(
            request.student_name,
            request.question_type,
            questions_from_db,
            answers_from_db,
            choices_from_db if request.question_type == "Multiple Choice" else None
        )
        
        record_data = {
            "student_name": request.student_name,
            "question_type": request.question_type,
            "results": evaluation_results_list
        }
        await examai.upsert_exam_record(record_data)

        return {"message": "Cevaplar başarıyla değerlendirildi ve kaydedildi.", "evaluation_result": evaluation_results_list}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevaplar değerlendirilirken veya kaydedilirken hata oluştu: {e}")

@app.post("/exam-record", summary="Yeni bir sınav kaydı oluşturur veya mevcut olanı günceller.")
async def create_or_update_exam_record_endpoint(
    request: CreateExamRecordRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        record_data = request.dict(exclude_unset=True)
        upserted_record = await examai.upsert_exam_record(record_data)
        
        if upserted_record:
            return {"message": "Sınav kaydı başarıyla oluşturuldu/güncellendi.", "record": upserted_record}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sınav kaydı oluşturulurken/güncellenirken beklenmeyen bir hata oluştu.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sınav kaydı oluşturulurken/güncellenirken hata oluştu: {e}")

@app.delete("/exam-record", summary="Belirli bir sınav kaydını siler.")
async def delete_exam_record_endpoint(
    request: DeleteExamRecordRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        # Supabase çağrısını run_in_threadpool ile sarmala
        response = await run_in_threadpool(lambda: examai.supabase.table('exam_records').delete().eq('student_name', request.student_name).eq('question_type', request.question_type).execute())
        if response.data:
            return {"message": "Sınav kaydı başarıyla silindi."}
        else:
            # Supabase'den "rows not found" hatası gelirse 404 döndür
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek sınav kaydı bulunamadı.")
    except HTTPException:
        # Eğer zaten bir HTTPException fırlatılmışsa, onu tekrar fırlat
        raise
    except Exception as e:
        # Genel hatalar için 500 döndür
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sınav kaydı silinirken hata oluştu: {e}")

@app.get("/record", summary="Belirli bir sınav kaydını döndürür.")
async def get_single_exam_record_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        record = await examai.get_student_exam_record(student_name, question_type)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sınav kaydı bulunamadı.")
        return {"student_name": student_name, "question_type": question_type, "record": record}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sınav kaydı alınırken hata oluştu: {e}")

@app.get("/question", summary="Belirli bir soruyu döndürür.")
async def get_single_question_endpoint(student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        question = await examai.get_question(student_name, question_type, index)
        if question is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru bulunamadı veya kayıt mevcut değil.")
        return {"question": question}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru alınırken hata oluştu: {e}")

@app.get("/questions", summary="Tüm soruları döndürür.")
async def get_all_questions_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        questions = await examai.get_questions_all(student_name, question_type)
        if questions is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru bulunamadı veya kayıt mevcut değil.")
        return {"questions": questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm sorular alınırken hata oluştu: {e}")

@app.get("/count", summary="Belirli bir öğrenci için belirtilen soru tipine göre soru sayısını döndürür.")
async def get_question_count_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        record = await examai.get_student_exam_record(student_name, question_type)
        if not record or 'questions' not in record:
            return {"student_name": student_name, "question_type": question_type, "question_count": 0, "message": f"{question_type} soru bulunamadı veya kayıt mevcut değil."}
        return {"question_count": len(record['questions'])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru sayısı alınırken hata oluştu: {e}")

@app.get("/choice", summary="Belirli bir çoktan seçmeli sorunun belirli bir seçeneğini döndürür.")
async def get_single_choice_endpoint(student_name: str, question_index: int, choice_index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        question_type = "Multiple Choice"
        choice = await examai.get_choice(student_name, question_type, question_index, choice_index)
        if choice is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seçenek bulunamadı veya kayıt mevcut değil.")
        return {"choice": choice}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Seçenek alınırken hata oluştu: {e}")

@app.get("/choices", summary="Belirli bir çoktan seçmeli sorunun tüm seçeneklerini döndürür.")
async def get_all_choices_for_single_question_endpoint(student_name: str, question_index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        question_type = "Multiple Choice"
        record = await examai.get_student_exam_record(student_name, question_type)
        if not record or 'choices' not in record or not record['choices']:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Çoktan seçmeli seçenekler bulunamadı veya kayıt mevcut değil.")

        if question_index < 0 or question_index >= len(record['choices']):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz soru indeksi.")

        return {"choices": record['choices'][question_index]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Belirli sorunun seçenekleri alınırken hata oluştu: {e}")

@app.get("/choices/all", summary="Belirli bir öğrencinin tüm çoktan seçmeli sorularının tüm seçeneklerini döndürür.")
async def get_all_choices_for_student_endpoint(student_name: str, _ = Depends(verify_castrumai_api_key)):
    try:
        # This endpoint is specifically for "Multiple Choice" questions.
        question_type = "Multiple Choice"
        record = await examai.get_student_exam_record(student_name, question_type)
        if not record or 'choices' not in record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Çoktan seçmeli seçenekler bulunamadı veya kayıt mevcut değil.")
        return {"choices": record['choices']}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm seçenekler alınırken hata oluştu: {e}")

@app.get("/answer", summary="Belirli bir cevabı döndürür.")
async def get_single_answer_endpoint(student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        answer = await examai.get_answer(student_name, question_type, index)
        if answer is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevap bulunamadı veya kayıt mevcut değil.")
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevap alınırken hata oluştu: {e}")

@app.get("/answers", summary="Tüm cevapları döndürür.")
async def get_all_answers_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        answers = await examai.get_answers_all(student_name, question_type)
        if answers is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevap bulunamadı veya kayıt mevcut değil.")
        return {"answers": answers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm cevaplar alınırken hata oluştu: {e}")

@app.get("/result", summary="Belirli bir sonucunu döndürür.")
async def get_single_result_endpoint(student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        result = await examai.get_result(student_name, question_type, index)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuç bulunamadı veya kayıt mevcut değil.")
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sonuç alınırken hata oluştu: {e}")

@app.get("/results", summary="Tüm sonuçları döndürür.")
async def get_all_results_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        results = await examai.get_results_all(student_name, question_type)
        if results is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuç bulunamadı veya kayıt mevcut değil.")
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm sonuçlar alınırken hata oluştu: {e}")

@app.get("/score", summary="Öğrencinin toplam puanını döndürür.")
async def get_total_score_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        score = await examai.get_total_score(student_name, question_type)
        if score is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Puan bulunamadı veya kayıt mevcut değil.")
        return {"total_score": score}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Toplam puan alınırken hata oluştu: {e}")

@app.get("/plagiarism-violations", summary="Öğrencinin intihal ihlallerini döndürür.")
async def get_plagiarism_violations_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        violations = await examai.get_plagiarism_violations(student_name, question_type)
        if violations is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="İntihal ihlalleri bulunamadı.")
        return {"plagiarism_violations": violations}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"İntihal ihlalleri alınırken hata oluştu: {e}")

@app.put("/update/question", summary="Belirli bir soruyu günceller.")
async def update_single_question_endpoint(
    request: UpdateQuestionValueRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_question_in_record(
            request.student_name,
            request.question_type,
            request.question_index,
            request.value
        )
        if updated_record:
            return {"message": "Soru başarıyla güncellendi.", "questions": updated_record.get('questions', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru güncellenirken hata oluştu: {e}")

@app.put("/update/questions/all", summary="Belirli bir öğrencinin tüm sorularını günceller.")
async def update_all_questions_endpoint(
    request: UpdateAllQuestionsRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_all_questions_in_record(
            request.student_name,
            request.question_type,
            request.questions
        )
        if updated_record:
            return {"message": "Tüm sorular başarıyla güncellendi.", "questions": updated_record.get('questions', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tüm sorular güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm sorular güncellenirken hata oluştu: {e}")

@app.put("/update/choice", summary="Çoktan seçmeli bir sorunun belirli bir seçeneğini günceller.")
async def update_choice_endpoint(
    request: UpdateChoiceRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_choice_in_record(
            request.student_name,
            "Multiple Choice",
            request.question_index,
            request.choice_index,
            request.value
        )
        if updated_record:
            return {"message": "Seçenek başarıyla güncellendi.", "choices": updated_record.get('choices', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seçenek güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Seçenek güncellenirken hata oluştu: {e}")

@app.put("/update/question/choices", summary="Çoktan seçmeli bir sorunun tüm seçeneklerini günceller.")
async def update_question_choices_endpoint(
    request: UpdateQuestionChoicesRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        question_type = "Multiple Choice"
        updated_record = await examai.update_choices_for_single_question_in_record(
            request.student_name,
            question_type,
            request.question_index,
            request.choices
        )
        if updated_record:
            return {"message": "Sorunun seçenekleri başarıyla güncellendi.", "choices": updated_record.get('choices', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sorunun seçenekleri güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sorunun seçenekleri güncellenirken hata oluştu: {e}")

@app.put("/update/choices/all", summary="Belirli bir öğrencinin tüm çoktan seçmeli sorularının tüm seçeneklerini günceller.")
async def update_all_choices_endpoint(
    request: UpdateAllChoicesRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        question_type = "Multiple Choice"
        updated_record = await examai.update_all_choices_in_record(
            request.student_name,
            question_type,
            request.choices
        )
        if updated_record:
            return {"message": "Tüm seçenekler başarıyla güncellendi.", "choices": updated_record.get('choices', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tüm seçenekler güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm seçenekler güncellenirken hata oluştu: {e}")

@app.put("/update/answer", summary="Belirli bir sorunun cevabını günceller.")
async def update_single_answer_endpoint(
    request: AnswerUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_answer(
            request.student_name,
            request.question_type,
            request.index,
            request.answer
        )
        if updated_record:
            return {"message": "Cevap başarıyla güncellendi.", "answers": updated_record.get('answers', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevap güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevap güncellenirken hata oluştu: {e}")

@app.put("/update/answers/bulk", summary="Tüm cevapları toplu olarak günceller.")
async def update_bulk_answers_endpoint(
    request: AnswersBulkUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_answers_bulk(
            request.student_name,
            request.question_type,
            request.answers
        )
        if updated_record:
            return {"message": "Cevaplar toplu olarak başarıyla güncellendi.", "answers": updated_record.get('answers', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevaplar toplu olarak güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevaplar toplu olarak güncellenirken hata oluştu: {e}")

@app.put("/update/result", summary="Belirli bir sorunun sonucunu günceller.")
async def update_single_result_endpoint(
    request: ResultUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_result(
            request.student_name,
            request.question_type,
            request.index,
            request.result
        )
        if updated_record:
            return {"message": "Sonuç başarıyla güncellendi.", "results": updated_record.get('results', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuç güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sonuç güncellenirken hata oluştu: {e}")

@app.put("/update/results/bulk", summary="Tüm sonuçları toplu olarak günceller.")
async def update_bulk_results_endpoint(
    request: ResultsBulkUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_results_bulk(
            request.student_name,
            request.question_type,
            request.results
        )
        if updated_record:
            return {"message": "Sonuçlar toplu olarak başarıyla güncellendi.", "results": updated_record.get('results', [])}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuçlar toplu olarak güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sonuçlar toplu olarak güncellenirken hata oluştu: {e}")

@app.put("/update/plagiarism-violation", summary="Öğrencinin intihal ihlali bilgisini ekler veya günceller.")
async def update_plagiarism_violation_endpoint(
    request: AddPlagiarismViolationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_plagiarism_violations_in_record(
            request.student_name,
            request.question_type,
            request.violation_text
        )
        if updated_record:
            return {"message": "İntihal ihlali başarıyla eklendi/güncellendi.", "plagiarism_violations": updated_record.get('plagiarism_violations', "")}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="İntihal ihlali eklenemedi/güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"İntihal ihlali eklenirken/güncellenirken hata oluştu: {e}")
