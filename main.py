import os
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import examai
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool

load_dotenv()

app = FastAPI(
    title="ExamAI API",
    description="Yeni nesil OpenAI asistanları ile güncellenmiş, sınav ve değerlendirme işlemlerini yürüten API.",
    version="1.0.0"
)

# --- API Anahtar Doğrulaması ---
CASTRUMAI_API_KEY_HEADER_NAME = "castrumai-apikey"
VALID_CASTRUMAI_API_KEY = os.getenv("CASTRUMAI_API_KEY")

if not VALID_CASTRUMAI_API_KEY:
    raise ValueError("CASTRUMAI_API_KEY ortam değişkeni ayarlanmamış.")

async def verify_castrumai_api_key(castrumai_apikey: Optional[str] = Header(None, alias=CASTRUMAI_API_KEY_HEADER_NAME)):
    if castrumai_apikey != VALID_CASTRUMAI_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz CastrumAI API Anahtarı")
    return True

# --- Pydantic Modelleri (Değişiklik yok) ---
class OpenEndedQuestionGenerationRequest(BaseModel):
    student_name: str
    number_of_questions: int

class MultipleChoiceQuestionGenerationRequest(BaseModel):
    student_name: str
    number_of_questions: int
    number_of_choices: int

class AnswerEvaluationRequest(BaseModel):
    student_name: str

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
    correct_answers: Optional[List[str]] = None
    answers: Optional[List[str]] = None
    results: Optional[List[str]] = None
    total_score: Optional[float] = 0.0
    plagiarism_violations: Optional[str] = ""

class DeleteExamRecordRequest(BaseModel):
    student_name: str
    question_type: str

class UpdateQuestionValueRequest(BaseModel):
    student_name: str
    question_type: str
    question_index: int
    question: str
    correct_answer: Optional[str] = None

class UpdateAllQuestionsRequest(BaseModel):
    student_name: str
    question_type: str
    questions: List[str]
    correct_answers: Optional[List[str]] = None

class UpdateChoiceRequest(BaseModel):
    student_name: str
    question_index: int
    choice_index: int
    value: str

class UpdateQuestionChoicesRequest(BaseModel):
    student_name: str
    question_index: int
    choices: List[str]

class UpdateAllChoicesRequest(BaseModel):
    student_name: str
    choices: List[List[str]]

class AddPlagiarismViolationRequest(BaseModel):
    student_name: str
    question_type: str
    violation_text: str

class UpdateCorrectAnswerRequest(BaseModel):
    student_name: str
    question_type: str
    index: int
    correct_answer: str

class UpdateAllCorrectAnswersRequest(BaseModel):
    student_name: str
    question_type: str
    correct_answers: List[str]

class DeleteSingleQuestionRequest(BaseModel):
    student_name: str
    question_type: str
    index: int

class DeleteAllQuestionsRequest(BaseModel):
    student_name: str
    question_type: str

# --- API Uç Noktaları ---

@app.post("/generate/open-ended", summary="AI ile açık uçlu sorular oluşturur veri tabanına ekler.")
async def generate_open_ended(
    request: OpenEndedQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        generated_data = await examai.generate_open_ended_questions_with_openai_assistant(
            request.number_of_questions
        )

        new_questions = generated_data.get("questions", [])
        new_correct_answers = generated_data.get("correct_answers", [])

        if not new_questions or not new_correct_answers:
            raise HTTPException(status_code=500, detail="Asistandan soru veya cevap verisi alınamadı.")

        existing_record = await examai.get_student_exam_record(request.student_name, "Open Ended")

        if existing_record:
            existing_questions = existing_record.get('questions') or []
            existing_correct_answers = existing_record.get('correct_answers') or []
            
            all_questions = existing_questions + new_questions
            all_correct_answers = existing_correct_answers + new_correct_answers
        else:
            all_questions = new_questions
            all_correct_answers = new_correct_answers

        record_data = {
            "student_name": request.student_name,
            "question_type": "Open Ended",
            "questions": all_questions,
            "correct_answers": all_correct_answers
        }
        
        await examai.upsert_exam_record(record_data)

        return new_questions

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Açık uçlu sorular üretilirken hata oluştu: {e}")


@app.post("/generate/mcq", summary="AI ile çoktan seçmeli soruları, şıkları ve doğru cevapları oluşturur ve veri tabanına ekler.")
async def generate_mcq(
    request: MultipleChoiceQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        existing_record = await examai.get_student_exam_record(request.student_name, "Multiple Choice")

        # --- YENİ KONTROL MEKANİZMASI ---
        if existing_record:
            existing_choices = existing_record.get('choices')
            # Eğer mevcut sınavda daha önceden eklenmiş şıklar varsa
            if existing_choices and len(existing_choices) > 0 and len(existing_choices[0]) > 0:
                # Mevcut şık sayısını al
                current_choice_count = len(existing_choices[0])
                # Yeni istenen şık sayısı ile karşılaştır
                if current_choice_count != request.number_of_choices:
                    # Eğer farklıysa, hata fırlat
                    raise HTTPException(
                        status_code=400, # Bad Request
                        detail=f"Mevcut sınavda her soru için {current_choice_count} şık bulunmaktadır. Farklı sayıda ({request.number_of_choices}) şıkka sahip yeni sorular ekleyemezsiniz. Lütfen aynı şık sayısını kullanın veya yeni bir sınav oluşturun."
                    )
        # --- KONTROL MEKANİZMASI SONU ---

        generated_data = await examai.generate_multiple_choice_questions_with_openai_assistant(
            request.number_of_questions,
            request.number_of_choices
        )
        
        new_questions = generated_data.get("questions", [])
        new_choices = generated_data.get("choices", [])
        new_correct_answers = generated_data.get("correct_answers", [])

        if not all([new_questions, new_choices, new_correct_answers]):
            raise HTTPException(status_code=500, detail="Asistandan eksik veya hatalı veri alındı.")

        if existing_record:
            existing_questions = existing_record.get('questions') or []
            existing_choices_list = existing_record.get('choices') or []
            existing_correct_answers = existing_record.get('correct_answers') or []

            all_questions = existing_questions + new_questions
            all_choices = existing_choices_list + new_choices
            all_correct_answers = existing_correct_answers + new_correct_answers
        else:
            all_questions = new_questions
            all_choices = new_choices
            all_correct_answers = new_correct_answers

        record_data = {
            "student_name": request.student_name,
            "question_type": "Multiple Choice",
            "questions": all_questions,
            "choices": all_choices,
            "correct_answers": all_correct_answers
        }
        
        await examai.upsert_exam_record(record_data)

        return generated_data

    except HTTPException as e:
        # Önceden fırlatılan HTTP hatalarını doğrudan gönder
        raise e
    except Exception as e:
        # Diğer tüm hatalar için genel bir hata mesajı göster
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Çoktan seçmeli sorular üretilirken beklenmeyen bir hata oluştu: {e}")


@app.post("/evaluate", summary="Verilen açık uçlu cevapları değerlendirir. Bu fonksiyonu kullanmadan önce update/answer veya update/answer/bulk fonksiyonları ile soru sayısı kadar cevap eklenmelidir.")
async def evaluate_answers(
    request: AnswerEvaluationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    question_type = "Open Ended"
    try:
        exam_record = await examai.get_student_exam_record(request.student_name, question_type)

        if not exam_record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Değerlendirme için sınav kaydı bulunamadı.")

        questions = exam_record.get('questions')
        correct_answers = exam_record.get('correct_answers')
        answers = exam_record.get('answers')

        if not all([questions, correct_answers, answers]):
            raise HTTPException(status_code=400, detail="Eksik cevap var.")
        
        if not (len(questions) == len(correct_answers) == len(answers)):
            raise HTTPException(status_code=400, detail="Soru sayısı ve öğrenci cevapları sayıları eşleşmiyor.")

        evaluation_results = await examai.check_answers_with_openai_assistant(
            questions=questions,
            correct_answers=correct_answers,
            answers=answers
        )
        
        exam_record['results'] = evaluation_results
        await examai.upsert_exam_record(exam_record)

        return evaluation_results
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevaplar değerlendirilirken hata oluştu: {e}")

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


@app.delete("/delete/question", summary="Belirtilen türdeki bir sınavdan belirli bir soruyu siler.")
async def delete_single_question_endpoint(
    request: DeleteSingleQuestionRequest,
    _ = Depends(verify_castrumai_api_key) # API anahtar doğrulamanız
):
    try:
        updated_record = await examai.delete_single_question(
            request.student_name,
            request.question_type,
            request.index
        )
        
        if updated_record is None:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadı.")

        return {"message": "Soru başarıyla silindi.", "updated_questions": updated_record.get('questions', [])}
    
    except ValueError as e:
        # examai'den gelen doğrulama hatalarını yakala
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru silinirken hata oluştu: {e}")

@app.delete("/delete/questions/all", summary="Belirtilen türdeki bir sınavdaki tüm soruları siler.")
async def delete_all_questions_endpoint(
    request: DeleteAllQuestionsRequest,
    _ = Depends(verify_castrumai_api_key) # API anahtar doğrulamanız
):
    try:
        updated_record = await examai.delete_all_questions(
            request.student_name,
            request.question_type
        )
        if updated_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadı.")

        return {"message": "Tüm sorular başarıyla silindi.", "updated_questions": updated_record.get('questions')} # None dönebilir
    
    except ValueError as e:
        # examai'den gelen doğrulama hatalarını yakala
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sorular silinirken hata oluştu: {e}")



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

@app.get("/correct-answer", summary="Belirli bir sorunun doğru cevabını döndürür.")
async def get_single_correct_answer_endpoint(student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        correct_answer = await examai.get_correct_answer(student_name, question_type, index)
        if correct_answer is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doğru cevap bulunamadı veya kayıt mevcut değil.")
        return {"correct_answer": correct_answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Doğru cevap alınırken hata oluştu: {e}")

@app.get("/correct-answers", summary="Bir sınavdaki tüm doğru cevapları döndürür.")
async def get_all_correct_answers_endpoint(student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        correct_answers = await examai.get_correct_answers_all(student_name, question_type)
        if correct_answers is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doğru cevaplar bulunamadı veya kayıt mevcut değil.")
        return {"correct_answers": correct_answers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm doğru cevaplar alınırken hata oluştu: {e}")



@app.get("/answer", summary="Belirli bir öğrenci cevabını döndürür.")
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

@app.get("/answers", summary="Tüm öğrenci cevaplarını döndürür.")
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

@app.put("/update/question", summary="Belirli bir soruyu ve isteğe bağlı olarak doğru cevabını günceller.")
async def update_single_question_endpoint(
    request: UpdateQuestionValueRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_question_in_record(
            request.student_name,
            request.question_type,
            request.question_index,
            request.question,
            request.correct_answer
        )
        if updated_record:
            return {
                "questions": updated_record.get('questions'),
                "correct_answers": updated_record.get('correct_answers')
            }
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru güncellenirken hata oluştu: {e}")

@app.put("/update/questions/all", summary="Belirli bir öğrencinin tüm sorularını ve isteğe bağlı olarak doğru cevaplarını günceller.")
async def update_all_questions_endpoint(
    request: UpdateAllQuestionsRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_all_questions_in_record(
            request.student_name,
            request.question_type,
            request.questions,
            request.correct_answers
        )
        if updated_record:
            return {
                "questions": updated_record.get('questions'),
                "correct_answers": updated_record.get('correct_answers')
            }
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tüm sorular güncellenemedi veya kayıt bulunamadı.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
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
            return {"message": "Seçenek başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Seçenek güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Seçenek güncellenirken hata oluştu: {e}")

@app.put("/update/question/choices", summary="Çoktan seçmeli bir sorunun tüm seçeneklerini günceller.")
async def update_question_choices_endpoint(
    request: UpdateQuestionChoicesRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_choices_for_single_question_in_record(
            request.student_name,
            "Multiple Choice",
            request.question_index,
            request.choices
        )
        if updated_record:
            return {"message": "Sorunun seçenekleri başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sorunun seçenekleri güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sorunun seçenekleri güncellenirken hata oluştu: {e}")

@app.put("/update/choices/all", summary="Tüm çoktan seçmeli soruların tüm seçeneklerini günceller.")
async def update_all_choices_endpoint(
    request: UpdateAllChoicesRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_all_choices_in_record(
            request.student_name,
            "Multiple Choice",
            request.choices
        )
        if updated_record:
            return {"message": "Tüm seçenekler başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tüm seçenekler güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm seçenekler güncellenirken hata oluştu: {e}")

@app.put("/update/correct-answer", summary="Belirli bir sorunun doğru cevabını günceller.")
async def update_single_correct_answer_endpoint(
    request: UpdateCorrectAnswerRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_correct_answer_in_record(
            request.student_name,
            request.question_type,
            request.index,
            request.correct_answer
        )
        if updated_record:
            return {"message": "Doğru cevap başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadığı için doğru cevap güncellenemedi.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Doğru cevap güncellenirken hata oluştu: {e}")

@app.put("/update/correct-answers/all", summary="Tüm doğru cevapları toplu olarak günceller.")
async def update_all_correct_answers_endpoint(
    request: UpdateAllCorrectAnswersRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_all_correct_answers_in_record(
            request.student_name,
            request.question_type,
            request.correct_answers
        )
        if updated_record:
            return {"message": "Tüm doğru cevaplar başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadığı için doğru cevaplar güncellenemedi.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm doğru cevaplar güncellenirken hata oluştu: {e}")


@app.put("/update/answer", summary="Belirli bir sorunun öğrenci tarafından verilen cevabını günceller.")
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

@app.put("/update/answers/bulk", summary="Tüm öğrenci cevaplarını toplu olarak günceller.")
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


