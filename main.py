import os
from fastapi import FastAPI, HTTPException, Header, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import examai
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Header, status, Depends, UploadFile, File
from contextlib import asynccontextmanager
import json

load_dotenv()

# --- BU YENİ FONKSİYONU main.py DOSYANIZA EKLEYİN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlangıcında çalışacak kod
    print("Uygulama başlıyor, embedding önbelleği oluşturulacak...")
    await examai.initialize_file_name_embeddings()
    yield
    # Uygulama kapanırken çalışacak kod (şimdilik boş)
    print("Uygulama kapanıyor...")

app = FastAPI(
    lifespan=lifespan,
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

# --- Pydantic Modelleri (exam_name EKLENDİ) ---
class OpenEndedQuestionGenerationRequest(BaseModel):
    exam_name: str
    student_name: str
    number_of_questions: int
    question_topic: str

class MultipleChoiceQuestionGenerationRequest(BaseModel):
    exam_name: str
    student_name: str
    number_of_questions: int
    number_of_choices: int
    question_topic: str

class AnswerEvaluationRequest(BaseModel):
    exam_name: str
    student_name: str
    question_topic: str

class AnswerUpdateRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    index: int
    answer: str

class AnswersBulkUpdateRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    answers: List[str]

class ResultUpdateRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    index: int
    result: str

class ResultsBulkUpdateRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    results: List[str]

class CreateExamRecordRequest(BaseModel):
    exam_name: str
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
    exam_name: str
    student_name: str
    question_type: str

class UpdateQuestionValueRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    question_index: int
    question: str
    correct_answer: Optional[str] = None

class UpdateAllQuestionsRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    questions: List[str]
    correct_answers: Optional[List[str]] = None

class UpdateChoiceRequest(BaseModel):
    exam_name: str
    student_name: str
    question_index: int
    choice_index: int
    value: str

class UpdateQuestionChoicesRequest(BaseModel):
    exam_name: str
    student_name: str
    question_index: int
    choices: List[str]

class UpdateAllChoicesRequest(BaseModel):
    exam_name: str
    student_name: str
    choices: List[List[str]]

class AddPlagiarismViolationRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    violation_text: str

class UpdateCorrectAnswerRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    index: int
    correct_answer: str

class UpdateAllCorrectAnswersRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    correct_answers: List[str]

class DeleteSingleQuestionRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str
    index: int

class DeleteAllQuestionsRequest(BaseModel):
    exam_name: str
    student_name: str
    question_type: str

# --- Sözel Soru Üretme Modelleri ---

class VerbalQuestionRequest(BaseModel):
    exam_name: str
    student_name: str
    number_of_questions: int
    question_topic: str

class VerbalQuestionResponse(BaseModel):
    questions: List[str]
    feedback_guides: List[str] # Bu, veritabanına 'correct_answers' olarak kaydedilecek

# --- Sözel Geri Bildirim Modelleri ---

class VerbalFeedbackRequest(BaseModel):
    exam_name: str
    student_name: str
    
class VerbalFeedbackResponse(BaseModel):
    feedbacks: List[str]


# main.py dosyanıza bu yeni endpoint'i geçici olarak ekleyin

@app.get("/debug-record/{exam_name}/{student_name}", summary="Belirli bir sınav kaydındaki listelerin uzunluğunu kontrol eder.")
async def debug_exam_record(exam_name: str, student_name: str):
    """
    Bu endpoint, veritabanındaki belirli bir sınav kaydını çeker ve içindeki
    ana listelerin (questions, question_topics, vb.) tiplerini ve eleman sayılarını
    döndürür. "Veri tutarsızlığı" hatasının kaynağını bulmak için kullanılır.
    """
    question_type_to_use = "Open Ended"
    print(f"--- DEBUGGING RECORD for {exam_name} / {student_name} ---")
    
    try:
        exam_record = await examai.get_student_exam_record(exam_name, student_name, question_type_to_use)

        if not exam_record:
            print("HATA: Sınav kaydı bulunamadı.")
            raise HTTPException(status_code=404, detail="Sınav kaydı bulunamadı.")

        # Veritabanından gelen her bir listeyi al ve analiz et
        question_texts = exam_record.get('questions')
        question_topics = exam_record.get('question_topics')
        evaluation_rubrics = exam_record.get('evaluation_rubrics')
        answers = exam_record.get('answers')

        # Her bir listenin tipini ve uzunluğunu içeren bir rapor oluştur
        debug_info = {
            "message": "Veritabanından çekilen verinin analizi aşağıdadır. 'length' değerlerinin hepsi aynı olmalıdır.",
            "exam_name": exam_name,
            "student_name": student_name,
            "questions_list": {
                "type": str(type(question_texts)),
                "length": len(question_texts) if isinstance(question_texts, list) else None
            },
            "question_topics_list": {
                "type": str(type(question_topics)),
                "length": len(question_topics) if isinstance(question_topics, list) else None
            },
            "evaluation_rubrics_list": {
                "type": str(type(evaluation_rubrics)),
                "length": len(evaluation_rubrics) if isinstance(evaluation_rubrics, list) else None
            },
            "answers_list": {
                "type": str(type(answers)),
                "length": len(answers) if isinstance(answers, list) else None
            }
        }
        
        print("--- VERİTABANINDAN ÇEKİLEN VERİ ANALİZİ ---")
        print(json.dumps(debug_info, indent=2, ensure_ascii=False))
        print("-----------------------------------------")

        # Bu raporu hem konsola yazdır hem de kullanıcıya döndür
        return debug_info
        
    except Exception as e:
        print(f"Debug endpoint'inde hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- API Uç Noktaları ---
# main.py dosyanızdaki endpoint'ler

@app.post("/generate/open-ended", summary="AI ile açık uçlu sorular ve Rubric'ler oluşturur, veri tabanına ekler.")
async def generate_open_ended_with_rubrics(
    request: OpenEndedQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    question_type_to_use = "Open Ended"
    try:
        existing_record = await examai.get_student_exam_record(request.exam_name, request.student_name, question_type_to_use)
        
        # Veritabanında 'questions' (metin listesi) ve 'question_topics' (konu listesi) ayrı sütunlarda
        existing_question_texts = []
        if existing_record and isinstance(existing_record.get('questions'), list):
            existing_question_texts = existing_record['questions']
        
        existing_question_topics = []
        if existing_record and isinstance(existing_record.get('question_topics'), list):
            existing_question_topics = existing_record['question_topics']
            
        existing_evaluation_rubrics = [] 
        if existing_record and isinstance(existing_record.get('evaluation_rubrics'), list):
            existing_evaluation_rubrics = existing_record['evaluation_rubrics']

        # Üretim fonksiyonuna göndermek için mevcut veriyi birleştir
        existing_questions_for_prompt = [
            {"topic": topic, "question": text} 
            for topic, text in zip(existing_question_topics, existing_question_texts)
        ]

        generated_data = await examai.generate_open_ended_questions_with_rubrics_in_batch(
            request.number_of_questions,
            request.question_topic,
            existing_questions_for_prompt
        )

        new_questions_data = generated_data.get("questions", [])
        new_evaluation_rubrics = generated_data.get("evaluation_rubrics", [])

        if not new_questions_data or not new_evaluation_rubrics:
            raise HTTPException(status_code=500, detail="Modelden soru veya değerlendirme kriteri (rubric) verisi alınamadı.") 

        # Gelen veriyi veritabanı sütunları için ayır
        new_question_topics = [q.get('topic', 'Bilinmeyen Konu') for q in new_questions_data]
        new_question_texts = [q.get('question', 'Soru metni üretilemedi') for q in new_questions_data]

        # Tüm verileri birleştir
        all_question_texts = existing_question_texts + new_question_texts
        all_question_topics = existing_question_topics + new_question_topics
        all_evaluation_rubrics = existing_evaluation_rubrics + new_evaluation_rubrics
        
        record_data = {
            "exam_name": request.exam_name,
            "student_name": request.student_name,
            "question_type": question_type_to_use,
            "questions": all_question_texts,
            "question_topics": all_question_topics,
            "evaluation_rubrics": all_evaluation_rubrics
        }
        
        await examai.upsert_exam_record(record_data)

        return {"questions": new_questions_data}

    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Açık uçlu sorular ve rubric'ler üretilirken bir hata oluştu: {str(e)}")



@app.post("/evaluate", summary="Verilen açık uçlu cevapları Rubric sistemine göre değerlendirir.")
async def evaluate_answers_with_rubrics(
    request: AnswerEvaluationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    """
    Öğrenci cevaplarını, veritabanından çekilen yapılandırılmış Değerlendirme Kriterleri'ne (Rubric)
    göre değerlendirir ve sonuçları gerekçeleriyle birlikte kaydeder.
    """
    question_type_to_use = "Open Ended"
    try:
        # 1. Veritabanından sınav kaydını çek
        exam_record = await examai.get_student_exam_record(request.exam_name, request.student_name, question_type_to_use)

        if not exam_record:
            raise HTTPException(status_code=404, detail="Değerlendirme için sınav kaydı bulunamadı.")

        # 2. Gerekli verileri al ve doğrula
        question_texts = exam_record.get('questions')
        question_topics = exam_record.get('question_topics')
        evaluation_rubrics = exam_record.get('evaluation_rubrics')
        answers = exam_record.get('answers')
        
        if not all([question_texts, question_topics, evaluation_rubrics, answers]):
            raise HTTPException(
                status_code=400, 
                detail="Değerlendirme başlatılamadı: Sınavda eksik sorular, konular, rubric'ler veya öğrenci cevapları var."
            )
        
        if not (len(question_texts) == len(question_topics) == len(evaluation_rubrics) == len(answers)):
            error_detail = (
                f"Veri tutarsızlığı: Eleman sayıları eşleşmiyor. "
                f"Questions: {len(question_texts) if question_texts else 0}, "
                f"Topics: {len(question_topics) if question_topics else 0}, "
                f"Rubrics: {len(evaluation_rubrics) if evaluation_rubrics else 0}, "
                f"Answers: {len(answers) if answers else 0}."
            )
            raise HTTPException(status_code=400, detail=error_detail)

        # 3. `check_answers` fonksiyonunun beklediği formata getir
        questions_for_eval = [
            {"topic": topic, "question": text} 
            for topic, text in zip(question_topics, question_texts)
        ]

        # 4. Düzeltilmiş Rubric tabanlı cevap kontrol fonksiyonunu çağır
        evaluation_data = await examai.check_answers_in_batch_with_rubrics(
            questions_with_topics=questions_for_eval, # DÜZELTME: Doğru argüman adı kullanılıyor
            evaluation_rubrics=evaluation_rubrics,
            answers=answers
        )
        
        final_results = evaluation_data.get("results")
        final_reasonings = evaluation_data.get("reasonings")

        if not final_results or not final_reasonings or len(final_results) != len(question_texts):
             raise HTTPException(status_code=500, detail="AI'dan geçersiz veya eksik sayıda değerlendirme verisi alındı.")

        # 5. Sınav kaydını yeni sonuçlar ve gerekçelerle güncelle
        exam_record['results'] = final_results
        exam_record['reasonings'] = final_reasonings

        await examai.upsert_exam_record(exam_record)

        # 6. API yanıtını döndür
        return {
            "message": "Değerlendirme başarıyla tamamlandı.",
            "results": final_results,
            "reasonings": final_reasonings
        }
        
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"Cevaplar değerlendirilirken bir hata oluştu: {str(e)}")


@app.post("/generate/mcq", summary='AI ile çoktan seçmeli soruları ve şıkları oluşturur, veri tabanına ekler. Çoktan seçmeli sorular otomatik kontrol edilir. Cevapları harf olarak eklenmelidir örn. "a", "A", "b" benzeri')
async def generate_mcq(
    request: MultipleChoiceQuestionGenerationRequest,
    _ = Depends(verify_castrumai_api_key)
):
    # ... (Kontroller kısmı aynı kalacak) ...
        
    try:
        existing_record = await examai.get_student_exam_record(request.exam_name, request.student_name, "Multiple Choice")
        
        # Güvenli liste atamaları ve tip kontrolü
        existing_questions = []
        if existing_record and isinstance(existing_record.get('questions'), list):
            existing_questions = existing_record['questions']

        existing_choices = [] 
        if existing_record and isinstance(existing_record.get('choices'), list):
            existing_choices = existing_record['choices']

        existing_correct_answers = [] 
        if existing_record and isinstance(existing_record.get('correct_answers'), list):
            existing_correct_answers = existing_record['correct_answers']


        if existing_choices and len(existing_choices) > 0 and len(existing_choices[0]) > 0: 
            # mevcut_choices'ın list olduğu kontrolü yukarıda yapıldı
            current_choice_count = len(existing_choices[0])
            if current_choice_count != request.number_of_choices:
                raise HTTPException(
                    status_code=400,
                    detail=f"Mevcut sınavda her soru için {current_choice_count} şık bulunmaktadır. Farklı sayıda ({request.number_of_choices}) şıkka sahip yeni sorular ekleyemezsiniz. Lütfen aynı şık sayısını kullanın veya yeni bir sınav oluşturun."
                )

        generated_data = await examai.generate_multiple_choice_questions_in_batch(
            request.number_of_questions,
            request.number_of_choices,
            request.question_topic,
            existing_questions 
        )
        
        new_questions = generated_data.get("questions", [])
        if not isinstance(new_questions, list): 
            new_questions = []
        
        new_choices = generated_data.get("choices", [])
        if not isinstance(new_choices, list): 
            new_choices = []
        
        new_correct_answers = generated_data.get("correct_answers", [])
        if not isinstance(new_correct_answers, list): 
            new_correct_answers = []

        if not all([new_questions, new_choices, new_correct_answers]):
            raise HTTPException(status_code=500, detail="Modelden eksik veya hatalı veri alındı (boş liste döndü).") 

        all_questions = existing_questions + new_questions
        all_choices = existing_choices + new_choices
        all_correct_answers = existing_correct_answers + new_correct_answers
        
        record_data = {
            "exam_name": request.exam_name,
            "student_name": request.student_name,
            "question_type": "Multiple Choice",
            "questions": all_questions,
            "choices": all_choices,
            "correct_answers": all_correct_answers
        }
        
        await examai.upsert_exam_record(record_data)

        return generated_data

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Çoktan seçmeli sorular üretilirken beklenmeyen bir hata oluştu: {e}")



# --- YENİ ENDPOINT: Sözel Soru Üretme ---
@app.post("/generate/verbal", response_model=VerbalQuestionResponse)
async def generate_verbal_exam_questions(request: VerbalQuestionRequest, _ = Depends(verify_castrumai_api_key)):
    """
    Belirtilen konu hakkında, 1-2 dakikalık sözel cevap gerektiren sorular üretir.
    Ayrıca, bu cevapları değerlendirmek için bir insan eğitmene yardımcı olacak
    kapsamlı "geri bildirim rehberleri" oluşturur.
    """
    try:
        # Önceki sınavda aynı türden soru olup olmadığını kontrol et
        existing_questions = await examai.get_questions_all(
            exam_name=request.exam_name, 
            student_name=request.student_name, 
            question_type="Verbal Question"
        )

        # examai'dan yeni sözel soruları ve rehberleri üretmesini iste
        result = await examai.generate_verbal_questions(
            number_of_questions=request.number_of_questions,
            question_topic=request.question_topic,
            existing_questions=existing_questions
        )
        
        if not result or not result.get("questions"):
            raise HTTPException(status_code=500, detail="Modelden soru verisi alınamadı.")

        # Üretilen soruları ve rehberleri (correct_answers olarak) veritabanına kaydet
        await examai.update_all_questions_in_record(
            exam_name=request.exam_name,
            student_name=request.student_name,
            question_type="verbal",
            new_questions=result["questions"],
            new_correct_answers=result["correct_answers"] # Geri bildirim rehberleri buraya kaydediliyor
        )
        
        return VerbalQuestionResponse(
            questions=result["questions"],
            feedback_guides=result["correct_answers"]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Sözel soru üretilirken hata oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Sözel sorular üretilirken bir hata oluştu: {e}")


# --- YENİ ENDPOINT: Sözel Cevaplar İçin Geri Bildirim Üretme ---
@app.post("/feedback/verbal", response_model=VerbalFeedbackResponse)
async def get_feedback_for_verbal_answers(request: VerbalFeedbackRequest, _ = Depends(verify_castrumai_api_key)):
    """
    Veritabanında kayıtlı sözel sorular, geri bildirim rehberleri ve öğrenci cevaplarına
    dayanarak, bir insan eğitmene sunulmak üzere yapıcı geri bildirim metinleri üretir.
    """
    try:
        question_type = "Verbal Question" # question_type burada otomatik olarak ayarlandı.

        # Değerlendirme için gerekli tüm verileri veritabanından çek
        questions = await examai.get_questions_all(request.exam_name, request.student_name, question_type)
        feedback_guides = await examai.get_correct_answers_all(request.exam_name, request.student_name, question_type)
        student_answers = await examai.get_answers_all(request.exam_name, request.student_name, question_type)

        if not all([questions, feedback_guides, student_answers]):
            raise HTTPException(status_code=404, detail="Değerlendirme için gerekli veriler (sorular, rehberler veya cevaplar) bulunamadı.")
        
        if not (len(questions) == len(feedback_guides) == len(student_answers)):
            raise HTTPException(status_code=409, detail="Veritabanındaki veri tutarsızlığı: Soru, rehber ve cevap sayıları eşleşmiyor.")

        # examai'dan geri bildirimleri üretmesini iste
        feedbacks = await examai.provide_feedback_on_verbal_answers(
            questions=questions,
            feedback_guides=feedback_guides,
            student_verbal_answers=student_answers
        )

        if not feedbacks:
            raise HTTPException(status_code=500, detail="Modelden geri bildirim alınamadı.")

        return VerbalFeedbackResponse(feedbacks=feedbacks)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Sözel geri bildirim üretilirken hata oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Sözel geri bildirim üretilirken bir hata oluştu: {e}")

# main.py dosyanızdaki /evaluate endpoint'ini bununla değiştirin



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
        response = await run_in_threadpool(lambda: examai.supabase.table('exam_records')
                                           .delete()
                                           .eq('exam_name', request.exam_name)
                                           .eq('student_name', request.student_name)
                                           .eq('question_type', request.question_type)
                                           .execute())
        if response.data:
            return {"message": "Sınav kaydı başarıyla silindi."}
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Silinecek sınav kaydı bulunamadı.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sınav kaydı silinirken hata oluştu: {e}")


@app.delete("/delete/question", summary="Belirtilen türdeki bir sınavdan belirli bir soruyu siler.")
async def delete_single_question_endpoint(
    request: DeleteSingleQuestionRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.delete_single_question(
            request.exam_name,
            request.student_name,
            request.question_type,
            request.index
        )
        
        if updated_record is None:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadı.")

        return {"message": "Soru başarıyla silindi.", "updated_questions": updated_record.get('questions', [])}
    
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru silinirken hata oluştu: {e}")

@app.delete("/delete/questions/all", summary="Belirtilen türdeki bir sınavdaki tüm soruları siler.")
async def delete_all_questions_endpoint(
    request: DeleteAllQuestionsRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.delete_all_questions(
            request.exam_name,
            request.student_name,
            request.question_type
        )
        if updated_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadı.")

        return {"message": "Tüm sorular başarıyla silindi.", "updated_questions": updated_record.get('questions')}
    
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sorular silinirken hata oluştu: {e}")


@app.get("/record", summary="Belirli bir sınav kaydını döndürür.")
async def get_single_exam_record_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        record = await examai.get_student_exam_record(exam_name, student_name, question_type)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sınav kaydı bulunamadı.")
        return {"exam_name": exam_name, "student_name": student_name, "question_type": question_type, "record": record}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sınav kaydı alınırken hata oluştu: {e}")

@app.get("/question", summary="Belirli bir soruyu döndürür.")
async def get_single_question_endpoint(exam_name: str, student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        question = await examai.get_question(exam_name, student_name, question_type, index)
        if question is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru bulunamadı veya kayıt mevcut değil.")
        return {"question": question}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru alınırken hata oluştu: {e}")

@app.get("/questions", summary="Tüm soruları döndürür.")
async def get_all_questions_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        questions = await examai.get_questions_all(exam_name, student_name, question_type)
        if questions is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soru bulunamadı veya kayıt mevcut değil.")
        return {"questions": questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm sorular alınırken hata oluştu: {e}")

@app.get("/count", summary="Belirli bir öğrenci için belirtilen sınav adı ve soru tipine göre soru sayısını döndürür.")
async def get_question_count_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        record = await examai.get_student_exam_record(exam_name, student_name, question_type)
        if not record or 'questions' not in record:
            return {"exam_name": exam_name, "student_name": student_name, "question_type": question_type, "question_count": 0, "message": f"{question_type} soru bulunamadı veya kayıt mevcut değil."}
        return {"question_count": len(record['questions'])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Soru sayısı alınırken hata oluştu: {e}")

@app.get("/correct-answer", summary="Belirli bir sorunun doğru cevabını döndürür.")
async def get_single_correct_answer_endpoint(exam_name: str, student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        correct_answer = await examai.get_correct_answer(exam_name, student_name, question_type, index)
        if correct_answer is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doğru cevap bulunamadı veya kayıt mevcut değil.")
        return {"correct_answer": correct_answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Doğru cevap alınırken hata oluştu: {e}")

@app.get("/correct-answers", summary="Bir sınavdaki tüm doğru cevapları döndürür.")
async def get_all_correct_answers_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        correct_answers = await examai.get_correct_answers_all(exam_name, student_name, question_type)
        if correct_answers is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doğru cevaplar bulunamadı veya kayıt mevcut değil.")
        return {"correct_answers": correct_answers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm doğru cevaplar alınırken hata oluştu: {e}")


@app.get("/answer", summary="Belirli bir öğrenci cevabını döndürür.")
async def get_single_answer_endpoint(exam_name: str, student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        answer = await examai.get_answer(exam_name, student_name, question_type, index)
        if answer is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevap bulunamadı veya kayıt mevcut değil.")
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cevap alınırken hata oluştu: {e}")

@app.get("/answers", summary="Tüm öğrenci cevaplarını döndürür.")
async def get_all_answers_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        answers = await examai.get_answers_all(exam_name, student_name, question_type)
        if answers is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cevap bulunamadı veya kayıt mevcut değil.")
        return {"answers": answers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm cevaplar alınırken hata oluştu: {e}")

@app.get("/result", summary="Belirli bir sonucunu döndürür.")
async def get_single_result_endpoint(exam_name: str, student_name: str, question_type: str, index: int, _ = Depends(verify_castrumai_api_key)):
    try:
        result = await examai.get_result(exam_name, student_name, question_type, index)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuç bulunamadı veya kayıt mevcut değil.")
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sonuç alınırken hata oluştu: {e}")

@app.get("/results", summary="Tüm sonuçları döndürür.")
async def get_all_results_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        results = await examai.get_results_all(exam_name, student_name, question_type)
        if results is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sonuç bulunamadı veya kayıt mevcut değil.")
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm sonuçlar alınırken hata oluştu: {e}")

@app.get("/score", summary="Öğrencinin toplam puanını döndürür.")
async def get_total_score_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        score = await examai.get_total_score(exam_name, student_name, question_type)
        if score is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Puan bulunamadı veya kayıt mevcut değil.")
        return {"total_score": score}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Toplam puan alınırken hata oluştu: {e}")

@app.get("/plagiarism-violations", summary="Öğrencinin intihal ihlallerini döndürür.")
async def get_plagiarism_violations_endpoint(exam_name: str, student_name: str, question_type: str, _ = Depends(verify_castrumai_api_key)):
    try:
        violations = await examai.get_plagiarism_violations(exam_name, student_name, question_type)
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
            request.exam_name,
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
            request.exam_name,
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
            request.exam_name,
            request.student_name,
            "Multiple Choice", # question_type burada sabit
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
            request.exam_name,
            request.student_name,
            "Multiple Choice", # question_type burada sabit
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
            request.exam_name,
            request.student_name,
            "Multiple Choice", # question_type burada sabit
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
            request.exam_name,
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
            request.exam_name,
            request.student_name,
            request.question_type,
            request.correct_answers
        )
        if updated_record:
            return {"message": "Tüm doğru cevaplar başarıyla güncellendi.", "record": updated_record}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kayıt bulunamadığı için doğru cevaplar güncellenemedi.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tüm doğru cevaplar güncellenirken hata oluştu: {e}")


@app.put("/update/answer", summary='Belirli bir sorunun öğrenci tarafından verilen cevabını günceller. Çoktan seçmeli sorular için cevaplar harf olarak eklenmelidir örn. "a", "A", "b" benzeri')
async def update_single_answer_endpoint(
    request: AnswerUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_answer(
            request.exam_name,
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

@app.put("/update/answers/bulk", summary='Tüm öğrenci cevaplarını toplu olarak günceller. Yeni cevaplar array olarak ["cevap 1", "cevap 2, ...] formatında eklenmelidir. Çoktan seçmeli sorular için ise ["a", "b", "a"] formatında harf olarak eklenmelidir.')
async def update_bulk_answers_endpoint(
    request: AnswersBulkUpdateRequest,
    _ = Depends(verify_castrumai_api_key)
):
    try:
        updated_record = await examai.update_answers_bulk(
            request.exam_name,
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
            request.exam_name,
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
            request.exam_name,
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
            request.exam_name,
            request.student_name,
            request.question_type,
            request.violation_text
        )
        if updated_record:
            return {"message": "İntihal ihlali başarıyla eklendi/güncellendi.", "plagiarism_violations": updated_record.get('plagiarism_violations', "")}
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="İntihal ihlali eklenemedi/güncellenemedi veya kayıt bulunamadı.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"İntihal ihlali eklenirken/güncellenirken hata oluştu: {e}")
    
# examai.py dosyasında, mevcut Supabase fonksiyonlarının (örn. get_student_exam_record) altına ekleyin

async def get_all_generated_questions_for_exam(exam_name: str, question_type: str) -> List[str]:
    """
    Belirli bir sınav adı ve soru tipi için, tüm öğrenciler tarafından üretilmiş
    tüm soruları veritabanından çeker ve tek bir liste olarak döndürür.
    """
    try:
        response = await run_in_threadpool(
            lambda: supabase.from_('exam_records') # Doğru tablo adı 'exam_records'
            .select('questions') # Sadece 'questions' sütununu çek
            .eq('exam_name', exam_name)
            .eq('question_type', question_type)
            .execute()
        )
        
        all_questions_across_students = []
        if response.data:
            for record in response.data:
                questions_list = record.get('questions')
                if isinstance(questions_list, list):
                    all_questions_across_students.extend(questions_list)
        
        return all_questions_across_students
    except Exception as e:
        print(f"Tüm sınav soruları çekilirken hata oluştu: {e}")
        # Hata durumunda boş liste dön, böylece model yine de soru üretebilir
        return []
    

# --- NEW ENDPOINT FOR VOICE ANSWERS ---

@app.post("/answers/voice", summary="Öğrenci sesli cevabını alır, metne çevirir ve veri tabanına kaydeder. question_type 'Verbal Question' olmalıdır.")
async def add_voice_answer_endpoint(
    exam_name: str,
    student_name: str,
    index: int, # Cevabın kaydedileceği sorunun indeksi
    file: UploadFile = File(..., description="Ses dosyası (örn: .mp3, .wav, .m4a)"),
    _ = Depends(verify_castrumai_api_key)
):
    """
    Belirtilen sınav, öğrenci ve indeks için sesli bir cevabı alır,
    OpenAI Whisper kullanarak metne çevirir ve ardından veri tabanındaki
    ilgili 'Verbal Question' tipindeki kayda ekler.
    """
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-m4a", "audio/mp4", "audio/webm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Desteklenmeyen dosya türü: {file.content_type}. Lütfen mp3, wav, m4a, mp4, webm gibi bir ses dosyası yükleyin."
        )

    try:
        # Ses dosyasını bellek içi bir File objesine dönüştürüyoruz
        # OpenAI API, dosya objesi bekliyor.
        import io
        audio_bytes = await file.read()
        audio_file_like_object = io.BytesIO(audio_bytes)
        audio_file_like_object.name = file.filename # Dosya adını korumak önemli

        # examai modülündeki yeni fonksiyonu çağırıyoruz
        transcribed_text = await examai.add_voice_answer(
            exam_name=exam_name,
            student_name=student_name,
            index=index,
            audio_file=audio_file_like_object # Bellek içi dosya objesini gönder
        )

        return {
            "message": "Sesli cevap başarıyla kaydedildi.",
            "transcribed_text": transcribed_text
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sesli cevap işlenirken hata oluştu: {e}")

# --- end of new endpoint ---
