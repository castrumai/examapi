import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import json
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Header, status, Depends, UploadFile, File
import math

import pypdf 
import tiktoken 
import random # Random import'u da buraya taşındı

load_dotenv()
OPENAI_ASSISTANT_ID_ANSWER_CHECKER = os.getenv("OPENAI_ASSISTANT_ID_ANSWER_CHECKER")

# --- Ortam Değişkenleri ve Konfigürasyon ---
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Kontroller ---
if not SUPABASE_URL: raise ValueError("SUPABASE_URL ortam değişkeni ayarlanmamış.")
if not SUPABASE_KEY: raise ValueError("SUPABASE_KEY (SUPABASE_ANON_KEY) ortam değişkeni ayarlanmamış.")
if not OPENAI_API_KEY: raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmamış.")

# --- İstemci Başlatma ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Modül Dosyaları ve Kök Dizin ---
PDF_BASE_PATH = os.getenv("PDF_BASE_PATH", "./pdfs") 

MODULE_FILES = {
    "M1": [
        "Launching Appliances Final.pdf",
        "Limit Switch & Fall Wires Types Final.pdf",
        "Release Gear Types Final.pdf",
        "Special Equipments & Tool Types Final.pdf",
        "Survival Craft Engine Types Final.pdf",
        "Survival Craft Types Final.pdf",
        "Winches Final.pdf"
    ],
    "M2": [
        "A-Frame Davit & Fast Rescue Boat Annual Inspections.pdf",
        "Accumulator Control & Refilling Final.pdf",
        "Brake Disassembly & Assembly Operation Final.pdf",
        "Conventional Lifeboat & Davit Annual Inspections Final.pdf",
        "Conventional Lifeboat & Freefall Boat Release Hook Test.pdf",
        "Fast Rescue Boat Hook Overhaul Final.pdf",
        "Freefall Boat & Davit Annual Inspections Final.pdf",
        "Freefall Boat Hook Overhaul Final.pdf",
        "Hydrostatic Interlock Diaphragm Control Final.pdf",
        "Release Cable Adjusting & Timing Set Up Final.pdf",
        "Release Mechanism Overhaul Final.pdf",
        "Rescue Boat & Davit Annual Inspection Final.pdf"
    ],
    "M3": [
        "5 Yearly Inspection for Conventional Lifeboat & Gravity Davit.pdf",
        "5 Yearly Inspection for Freefall Boat & Freefall Davit.pdf",
        "5 Yearly Inspection for Rescue Boat & Rescue Davit.pdf",
        "Load Test Calculation .pdf",
        "Load Test for Conentional Lifeboat & Gravity Davit.pdf",
        "Load Test Procedures for FFB + FFD + LB + LBD + RB + RBD Davit.pdf"
    ]
}

# --- Dosya Adından Modül ID'sine Eşleme ve Tersine Lookup Map (Case-Insensitive için) ---
FILE_TO_MODULE_MAP = {} 
FILE_LOOKUP_MAP = {} 
for mod_id, files_list in MODULE_FILES.items():
    for fname in files_list:
        FILE_TO_MODULE_MAP[fname.upper()] = mod_id
        FILE_LOOKUP_MAP[fname.upper()] = fname 

FILE_NAME_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}


# --- Modül Bazlı Konu Listeleri (Kapsamlı) ---
MODULE_TOPICS = {
    "M1": [
        "Launching Appliance Classifications (Single-Point Suspension)",
        "Launching Appliance Classifications (Two-Point Suspension)",
        "Launching Appliance Classifications (Free-Fall Launch)",
        "Winch Types (Single Drum, Twin Drum, Electric, Hydraulic)",
        "Limit Switch Types (Rotating Spindle, Position, Magnetic) and Functional Roles",
        "Fall Wire Types (Standard Wire Rope, Wedge Socket) and Characteristics",
        "Release Gear Types (On-Load, Off-Load, Combined, Free-Fall Hydraulic)",
        "Survival Craft Types (Lifeboats: Open, Partially Enclosed, Totally Enclosed, Freefall)",
        "Survival Craft Types (Rescue Boats: Rigid, Semi-Rigid Inflatable, Fast)",
        "Survival Craft Engine Types (Inboard, Outboard, Jet Propulsion) and Key Features",
        "Special Tools & Equipment Classification (Hydraulic Power Tools, Mechanical Tools, Diagnostic Tools)",
        "Hydraulic Hand Pumps and Nitrogen Charging Kits",
        "Hook Engagement Indicators and Reset Status Gauges"
    ],
    "M2": [
        "Annual Inspection Protocol (General Purpose and Scope)",
        "Davit Frame and Structure Inspection (Annual)",
        "Hoisting Mechanism Inspection (Annual)",
        "Winch and Brake Assembly Review (Annual)",
        "Brake Disassembly and Assembly Operation",
        "Fall Wire and Sheave Maintenance (Annual, Repositioning, Replacement)",
        "Remote Control and Electrical Components Inspection (Annual)",
        "Hook Mechanism and Release Gear Inspection (Annual FRB)",
        "Fast Rescue Boat Hook Overhaul Procedure",
        "Hydraulic System Inspection (Annual, Davit/Boat Specifics)",
        "Electrical System Checks (Annual, Boat Specifics)",
        "Safety Equipment and Emergency Supplies Inspection (Annual)",
        "Accumulator Control and Refilling Procedure",
        "Hydrostatic Interlock Diaphragm Control",
        "Release Cable Adjusting and Timing Set Up",
        "Release Mechanism Overhaul Procedure (General)",
        "Rescue Boat Annual Inspection Protocol (General)",
        "Rescue Davit Annual Inspection Protocol (General)",
        "Rescue Boat Engine and Propulsion Unit Inspection (Annual)",
        "Rescue Boat Fuel Storage and Delivery System Inspection (Annual)",
        "Rescue Boat Steering System Inspection (Annual)",
        "Lifeboat Outfitting and Safety Equipment Inspection (Annual)",
        "Documentation and Certification Review (Annual Inspections)"
    ],
    "M3": [
        "5-Yearly Inspection Purpose (General)",
        "Load Test Calculation Methodology (Formula, Principles)",
        "Load Test Procedure (Conventional Lifeboat & Gravity Davit)",
        "Load Test Procedure (Freefall Boat & Davit)",
        "Load Test Procedure (Rescue Boat & Davit)",
        "Load Test Safety Considerations (Personnel, PPE, Hazards)",
        "Load Test Documentation and Reporting",
        "Load Test Acceptance Criteria (Structural, Brake, Hydraulic)",
        "5-Yearly Structural Integrity Inspection (Lifeboats/Davits)",
        "5-Yearly Mechanical Systems Inspection (Lifeboats/Davits)",
        "5-Yearly Hydraulic Components Inspection (Lifeboats/Davits)",
        "5-Yearly Safety Fixtures Inspection (Lifeboats/Davits)",
        "5-Yearly Rigging & Cables Inspection (Lifeboats/Davits)",
        "5-Yearly Fastening & Supports Inspection (Lifeboats/Davits)",
        "5-Yearly Winch and Drum Assembly Inspection",
        "5-Yearly Fall Wire System Inspection",
        "5-Yearly Electrical System Inspection",
        "5-Yearly Hook Release Systems Inspection (Freefall)",
        "5-Yearly Engine and Propulsion Unit Inspection (Freefall)", 
        "5-Yearly Fuel Storage and Delivery System Inspection (Freefall)", 
        "5-Yearly Steering System Inspection (Freefall)", 
        "5-Yearly Rescue Boat Inspection Overview",
        "5-Yearly Rescue Davit Inspection Overview",
        "Common Inspection Challenges (Corrosion, Wear, Fatigue)"
    ]
}


# --- Yardımcı Fonksiyonlar ---

# --- Bu yeni fonksiyonu Yardımcı Fonksiyonlar bölümüne ekleyin ---
async def initialize_file_name_embeddings():
    """
    Uygulama başlangıcında tüm dosya adlarının embedding'lerini oluşturur ve önbelleğe alır.
    """
    print("--- Embedding önbelleği oluşturuluyor... ---")
    file_names_to_embed = list(FILE_LOOKUP_MAP.keys())
    
    try:
        # Toplu halde embedding isteği gönder
        response = await client.embeddings.create(
            input=file_names_to_embed,
            model="text-embedding-3-small"
        )
        
        for i, fname_upper in enumerate(file_names_to_embed):
            FILE_NAME_EMBEDDINGS_CACHE[fname_upper] = response.data[i].embedding
            
        print(f"--- {len(FILE_NAME_EMBEDDINGS_CACHE)} adet dosya adı için embedding önbelleği başarıyla oluşturuldu. ---")
    
    except Exception as e:
        print(f"HATA: Embedding önbelleği oluşturulurken kritik bir hata oluştu: {e}")
        # Bu kritik bir hata olduğu için uygulamayı durdurabilir veya hatayı loglayabilirsiniz.
        raise e

async def _run_openai_assistant(assistant_id: str, user_message_content: str) -> str:
    try:
        # Her çağrı için yeni bir thread oluşturulur
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message_content,
        )

        run = await client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        if run.status == 'completed':
            messages = await client.beta.threads.messages.list(thread_id=thread.id)
            # Asistanın son mesajını bul
            for message in messages.data:
                if message.role == "assistant":
                    for content_block in message.content:
                        if content_block.type == "text":
                            return content_block.text.value.strip()
            raise ValueError("Asistandan geçerli bir yanıt alınamadı.")
        else:
            raise ValueError(f"Asistan görevi tamamlanamadı. Durum: {run.status}")
    except Exception as e:
        print(f"OpenAI Asistan çalıştırılırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Asistan yanıt veremedi veya bir hata oluştu: {e}")

async def _call_openai_chat_model(system_message_content: str, user_message_content: str) -> str:
    print("chat model called")
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.7, 
            top_p=1.0,       
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Chat modeli çalıştırılırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Chat modeli yanıt veremedi veya bir hata oluştu: {e}")


async def _call_openai_nano_model_json(system_message_content: str, user_message_content: str) -> str:
    """
    gpt-4o-mini modelini JSON çıktısı bekleyerek çağıran fonksiyon.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.5, # Yaratıcılık ve tutarlılık arasında bir denge
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Nano modeli (JSON) çalıştırılırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Nano (JSON) modeli yanıt veremedi: {e}")


async def _call_openai_nano_model_text(system_message_content: str, user_message_content: str) -> str:
    """
    gpt-4o-mini modelini düz metin çıktısı bekleyerek çağıran fonksiyon.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content}
            ],
            temperature=0.7, # Feedback için daha doğal bir dil
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Nano modeli (Text) çalıştırılırken hata: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Nano (Text) modeli yanıt veremedi: {e}")

async def _get_embedding(text: str) -> List[float]:
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding oluşturulurken hata: {e}")
        raise HTTPException(status_code=500, detail=f"Metin embedding'i oluşturulamadı: {e}")

# YENİ: Anahtar kelimeyle alakalı dosyaları bulan semantik arama fonksiyonu
async def _find_relevant_files_by_keyword(keyword_query: str, top_n_files: int = 5) -> List[str]:
    """
    Bir anahtar kelime sorgusuna göre, önbelleğe alınmış dosya adı embedding'lerini kullanarak
    en alakalı dosyaları semantik olarak bulur.
    """
    if not keyword_query:
        return []
    
    # Sadece kullanıcının sorgusu için embedding oluştur
    query_embedding = await _get_embedding(keyword_query)
    
    file_name_similarities = []
    # Artık her dosya adı için API çağırmak yerine önbellekten okuyoruz
    for fname_upper, file_name_embedding in FILE_NAME_EMBEDDINGS_CACHE.items():
        similarity_score = sum(q * f for q, f in zip(query_embedding, file_name_embedding))
        
        # Orijinal dosya adını FILE_LOOKUP_MAP'ten alıyoruz
        original_file_name = FILE_LOOKUP_MAP.get(fname_upper)
        if original_file_name:
            file_name_similarities.append({"file_name": original_file_name, "similarity": similarity_score})
    
    file_name_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    found_files = [item['file_name'] for item in file_name_similarities if item['similarity'] > 0.4][:top_n_files]
            
    return found_files


# _retrieve_relevant_chunks fonksiyonu güncellendi: module_ids listesi ve file_names alacak
# SQL'deki LIMIT kaldırıldığı için tüm eşleşenler dönecek
async def _retrieve_relevant_chunks(query_text: str, module_ids: Optional[List[str]] = None, file_names: Optional[List[str]] = None, top_k: int = 50) -> List[Dict[str, Any]]: 
    """
    Kullanıcı sorgusuna ve (isteğe bağlı) modül ID'leri listesi/dosya adları listesine göre Supabase'den en alakalı metin parçalarını çeker.
    SQL'den tüm eşleşen parçaları çeker (LIMIT kaldırıldı).
    """
    try:
        print(f"\n--- _retrieve_relevant_chunks başladı, query_text: '{query_text}', module_ids: '{module_ids}', file_names: '{file_names}' ---")

        query_embedding = await _get_embedding(query_text)
        print(f"--- Embedding alındı, boyutu: {len(query_embedding)} ---")

        # Dinamik match_threshold belirleniyor
        current_match_threshold = 0.7 # Varsayılan olarak yüksek (genel arama için)
        if (module_ids and len(module_ids) > 0) or (file_names and len(file_names) > 0):
            # Eğer modül veya dosya filtresi varsa, eşiği düşür (daha fazla parça çekmek için)
            current_match_threshold = 0.2 # Bu değer daha önce 0.1 idi, testler için 0.2 iyi olabilir
            print(f"--- Dinamik Eşik: {current_match_threshold} (Modül/Dosya Filtresi Aktif) ---")
        else:
            print(f"--- Dinamik Eşik: {current_match_threshold} (Genel Arama) ---")


        rpc_args = { 
            'query_embedding': query_embedding,
            'match_threshold': current_match_threshold, # Dinamik eşik kullanılıyor
            'match_count': top_k # Bu değer RPC'ye hala gönderiliyor, ancak SQL'de LIMIT kaldırıldığı için sadece sıralama için kullanılır.
        }
        
        if module_ids and len(module_ids) > 0: 
            rpc_args['match_module_ids'] = [m.upper() for m in module_ids]
            print(f"--- RPC'ye Gönderilen match_module_ids: {rpc_args['match_module_ids']} ---")
        else:
            rpc_args['match_module_ids'] = None 
            print("--- module_ids filtresi uygulanmadı ---")
        
        if file_names and len(file_names) > 0:
            rpc_args['match_file_names'] = [f.upper() for f in file_names] 
            print(f"--- RPC'ye Gönderilen match_file_names: {rpc_args['match_file_names']} ---")
        else:
            rpc_args['match_file_names'] = None 
            print("--- file_names filtresi uygulanmadı ---")


        print("--- Supabase RPC çağrısı yapılıyor... ---")

        response = await run_in_threadpool(
            lambda: supabase.rpc('match_chunks', rpc_args).execute() 
        )
        
        print(f"\n--- Supabase'den Gelen Ham Yanıt Verisi (response.data) ---")
        print(response.data)
        print("-----------------------------------------------------------\n")
        
        if not response.data:
            return [] # Boş liste döndür
        
        return response.data # Doğrudan Dict listesi döndürüyoruz

    except Exception as e:
        print(f"\n--- _retrieve_relevant_chunks fonksiyonu içinde yakalanan hata ---")
        print(f"Hata detayı: {e}")
        print("--------------------------------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Bilgi çekme sırasında hata oluştu: {e}. Supabase RPC veya embedding servisini kontrol edin.")


# --- Soru Üretme Fonksiyonlarının Güncellenmesi (Her soru için ayrı çağrı ve dinamik konu/dosya seçimi) ---












# examai.py dosyanıza bu yeni fonksiyonu ekleyin

def _post_process_rubric(rubric: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Model tarafından üretilen ham rubriği alır ve sorunun doğasına göre
    mantıksal VE birleştirmesi ve formatlama yaparak onu mekanik denetime
    %100 uygun hale getirir.
    """
    processed_rubric = {
        "anahtar_kavram": rubric.get("anahtar_kavram", "N/A"),
        "kabul_kriterleri": [],
        "ret_kriterleri": []
    }

    completeness_keywords = [
        'types', 'classifications', 'categories', 'components',
        'differences', 'features', 'roles', 'methods', 'kinds'
    ]
    question_lower = question.lower()
    requires_completeness_check = any(keyword in question_lower for keyword in completeness_keywords)

    kabul_kriterleri_raw = rubric.get("kabul_kriterleri", [])
    if kabul_kriterleri_raw and isinstance(kabul_kriterleri_raw, list):
        formatted_criteria = [f"Cevap, '{str(k).strip()}' ifadesini içerir." for k in kabul_kriterleri_raw]
        
        if requires_completeness_check and len(formatted_criteria) > 1:
            joined_criteria = " VE ".join([f"({c})" for c in formatted_criteria])
            processed_rubric["kabul_kriterleri"] = [joined_criteria]
        else:
            processed_rubric["kabul_kriterleri"] = formatted_criteria

    ret_kriterleri_raw = rubric.get("ret_kriterleri", [])
    if ret_kriterleri_raw and isinstance(ret_kriterleri_raw, list):
        processed_rubric["ret_kriterleri"] = [
            f"Cevap, '{str(r).strip()}' ifadesini içerir." for r in ret_kriterleri_raw
        ]

    return processed_rubric


# --- ANA FONKSİYON (NİHAİ PROMPT VE AKILLI POST-PROCESSING İLE) ---

async def generate_open_ended_questions_with_rubrics_in_batch(
    number_of_questions: int,
    question_topic: str,
    existing_questions: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    İstenen sayıda soruyu ve rubriği üretir. Modelin görevi en iyi ham
    maddeleri seçmektir; kod ise bu maddeleri kusursuz bir mantıksal yapıya
    dönüştürür.
    """
    if number_of_questions <= 0:
        return {"questions": [], "evaluation_rubrics": []}

    # --- Konu ve Metin Hazırlığı (Değişiklik yok) ---
    retrieval_query_text = question_topic
    target_module_ids = []
    target_file_names = []
    if question_topic.upper() in MODULE_FILES:
        target_module_ids = [question_topic.upper()]
    elif question_topic.upper() in FILE_LOOKUP_MAP:
        target_file_names = [FILE_LOOKUP_MAP[question_topic.upper()]]
        target_module_ids = [FILE_TO_MODULE_MAP[question_topic.upper()]]
    elif ',' in question_topic:
        parts = [m.strip().upper() for m in question_topic.split(',')]
        for part in parts:
            if part not in MODULE_FILES:
                raise HTTPException(status_code=400, detail=f"Geçersiz modül ID'si: '{part}'.")
            target_module_ids.append(part)
    else:
        found_files_by_keyword = await _find_relevant_files_by_keyword(question_topic, top_n_files=5)
        if found_files_by_keyword:
            target_file_names = found_files_by_keyword
            for fname_original in found_files_by_keyword:
                mod_id = FILE_TO_MODULE_MAP.get(fname_original.upper())
                if mod_id and mod_id not in target_module_ids:
                    target_module_ids.append(mod_id)
        else:
            raise HTTPException(status_code=400, detail=f"Geçersiz konu/modül/dosya: '{question_topic}'.")

    available_topics_for_selection = []
    if target_module_ids:
        for mod_id in target_module_ids:
            if mod_id in MODULE_TOPICS:
                available_topics_for_selection.extend(MODULE_TOPICS[mod_id])
    
    if not available_topics_for_selection:
        raise HTTPException(status_code=404, detail=f"'{question_topic}' ile ilişkili konu bulunamadı.")
    
    all_topics_for_generation = []
    if number_of_questions > len(available_topics_for_selection):
        all_topics_for_generation.extend(available_topics_for_selection * (number_of_questions // len(available_topics_for_selection)))
        all_topics_for_generation.extend(random.sample(available_topics_for_selection, number_of_questions % len(available_topics_for_selection)))
    else:
        all_topics_for_generation = random.sample(available_topics_for_selection, number_of_questions)
    random.shuffle(all_topics_for_generation)

    all_retrieved_chunks_data = await _retrieve_relevant_chunks(retrieval_query_text, module_ids=target_module_ids, file_names=target_file_names, top_k=100)
    if not all_retrieved_chunks_data:
        raise HTTPException(status_code=404, detail="Bilgi kaynağında ilgili metin bulunamadı.")
        
    retrieval_content = ""
    for chunk_data in all_retrieved_chunks_data:
        retrieval_content += f"--- Kaynak: {chunk_data.get('file_name', 'Bilinmiyor')} ---\n{chunk_data['content']}\n\n"
    
    existing_questions_prompt_part = ""
    if existing_questions:
        existing_question_texts = [q['question'] for q in existing_questions if isinstance(q, dict) and 'question' in q]
        existing_questions_str = json.dumps(existing_question_texts, ensure_ascii=False)
        existing_questions_prompt_part = f"\nDAHA ÖNCE ÜRETİLMİŞ SORULAR (Bunlardan FARKLI sorular üretmelisin):\n{existing_questions_str}"

    # --- Batching Mantığı ve API Çağrıları ---
    tasks = []
    num_batches = 1
    if number_of_questions > batch_size:
        num_batches = math.ceil(number_of_questions / batch_size)
    
    topic_batches = [all_topics_for_generation[i::num_batches] for i in range(num_batches)]

    for i, topic_batch in enumerate(topic_batches):
        current_batch_size = len(topic_batch)
        if current_batch_size == 0: continue
        topics_list_str = json.dumps(topic_batch, ensure_ascii=False, indent=2)
        
        # --- NİHAİ, BASİTLEŞTİRİLMİŞ VE DÜZELTİLMİŞ PROMPT ---
        system_prompt = f"""
GÖREV VE KİŞİLİK:
Sen, bir "Rubric Derleyicisi (Compiler)" yapay zekasısın. Görevin, sana verilen knowledge_base metnini analiz etmek ve denetçi bir AI için yüksek kaliteli, ham değerlendirme verileri (rubric) üretmektir. Senin görevin, en isabetli ve spesifik kanıtları seçmektir.

TEMEL PRENSİPLER (DEĞİŞTİRİLEMEZ)
1. KANITA DAYALILIK: Seçtiğin her bir kriter, knowledge_base'den doğrudan alınabilir, spesifik bir kanıt (teknik terim, sayı, kural vb.) olmalıdır.
2. NETLİK: Seçtiğin kriterler, belirsiz veya yoruma açık olmamalıdır.
3. BAĞIMSIZLIK: Seçtiğin her kriter, kendi başına bir anlam ifade etmelidir.

ZORUNLU ÜRETİM SÜRECİ
Aşağıdaki adımları istisnasız ve belirtilen sırada YÜRÜT:

Adım 1: Soru Üretimi
topics_to_cover listesindeki her başlık için, toplamda {current_batch_size} adet olacak şekilde, knowledge_base'deki spesifik bilgileri sorgulayan, açık uçlu ve İngilizce sorular üret.

Adım 2: Ham Rubric Verisi Üretimi
Her soru için aşağıdaki yapıya harfiyen uyarak bir rubric elemanı oluştur:
- anahtar_kavram: Konunun en temel prensibini tek bir net cümleyle yaz.
- kabul_kriterleri: Soruya doğru cevap veren, birbirinden bağımsız, en ayırt edici 2-4 adet spesifik kanıtı bir LİSTE olarak yaz.
- ret_kriterleri: knowledge_base ile kanıtlanabilir şekilde çelişen veya yaygın bir yanlışı temsil eden 1-2 adet spesifik ifadeyi bir LİSTE olarak yaz.

NİHAİ UYARI: Görevin, karmaşık mantıksal yapılar (`VE`/`VEYA`) veya özel formatlar (`"Cevap, ... içerir"`) oluşturmak DEĞİLDİR. Sadece en kaliteli ve en doğru ham kanıtları (anahtar kelimeler, kısa ifadeler) listelemeye odaklan.

# --- HATA DÜZELTME: API'nin JSON modu için zorunlu olan bölüm eklendi ---
ZORUNLU ÇIKTI FORMATI (OUTPUT FORMAT REQUIREMENT)
Tüm çıktın, başka hiçbir metin olmadan, sadece ve sadece geçerli tek bir JSON nesnesi olmalıdır. Çıktının tamamı JSON formatında olmalıdır.
# --- HATA DÜZELTME SONU ---

HEDEFLENEN ÇIKTI ŞABLONU (Bu basit şablona %100 uy)
{{
  "questions": [
    {{
      "topic": "...",
      "question": "..."
    }}
  ],
  "evaluation_rubrics": [
    {{
      "anahtar_kavram": "Jet sevk sistemleri, farklı yataklar için spesifik yağlama yöntemleri kullanır ve korozyona karşı kurban anotlarla korunur.",
      "kabul_kriterleri": [
        "impeller draws water",
        "steering via waterjet redirection",
        "Forward bearing oil-lubricated",
        "Rear bearing grease-lubricated",
        "sacrificial anodes"
      ],
      "ret_kriterleri": [
        "jet propulsion sistemi için propeller kelimesi (doğrusu impeller)",
        "Forward bearing grease-lubricated (doğruları tam tersidir)"
      ]
    }}
  ]
}}
"""
        user_prompt = f"""
Aşağıdaki konu başlıklarının her biri için birer tane olmak üzere, toplamda {current_batch_size} adet soru ve her biri için bir Değerlendirme Kriteri (Rubric) üret:
Konu Listesi (`topics_to_cover`):
{topics_list_str}
{existing_questions_prompt_part}
"""
        tasks.append(_call_openai_chat_model(system_prompt, user_prompt))

    # --- Sonuçları Birleştirme ve AKILLI POST-PROCESSING ---
    final_questions = []
    final_evaluation_rubrics = []
    try:
        batch_responses = await asyncio.gather(*tasks)
        for i, response_text in enumerate(batch_responses):
            try:
                # Ham JSON çıktısını ayrıştır
                parsed_response = json.loads(response_text)
                batch_questions = parsed_response.get("questions", [])
                batch_rubrics_raw = parsed_response.get("evaluation_rubrics", [])

                # Gelen verilerin temel doğruluğunu kontrol et
                if not batch_questions or not batch_rubrics_raw or len(batch_questions) != len(topic_batches[i]) or len(batch_questions) != len(batch_rubrics_raw):
                    print(f"Batch {i+1} atlandı: Soru/Rubric sayısı eşleşmiyor veya veri eksik.")
                    continue

                # Her bir ham rubriği al ve sorusuyla birlikte işlemden geçir
                processed_rubrics_for_batch = []
                for idx, rubric_raw in enumerate(batch_rubrics_raw):
                    question_text = batch_questions[idx]['question']
                    processed_rubric = _post_process_rubric(rubric_raw, question_text)
                    processed_rubrics_for_batch.append(processed_rubric)
                
                final_questions.extend(batch_questions)
                final_evaluation_rubrics.extend(processed_rubrics_for_batch)

            except json.JSONDecodeError:
                print(f"Batch {i+1} atlandı: JSON ayrıştırma hatası.")
                continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toplu soru ve rubric üretimi sırasında bir hata oluştu: {str(e)}")

    # Sorular ve işlenmiş rubriklerin sırasının eşleştiğinden emin ol
    if len(final_questions) != len(final_evaluation_rubrics):
         raise HTTPException(status_code=500, detail="Son işleme sonrası soru ve rubric sayısı eşleşmiyor.")

    return {"questions": final_questions, "evaluation_rubrics": final_evaluation_rubrics}
















 # examai.py dosyanıza bu yeni fonksiyonu ekleyin

async def generate_multiple_choice_questions_in_batch(
    number_of_questions: int,
    number_of_choices: int,
    question_topic: str,
    existing_questions: Optional[List[str]] = None
) -> Dict[str, Any]:
    
    # --- 1. Adım: Konu ve Metin Parçacıklarını Hazırlama (Bu kısım aynı) ---
    
    # ... [Bu bölüm, generate_open_ended_questions_in_batch ile aynı:
    #      RAG ve konu seçimi kodları burada da geçerlidir.] ...
    
    retrieval_query_text = question_topic
    target_module_ids = []
    target_file_names = []

    if question_topic.upper() in MODULE_FILES:
        target_module_ids = [question_topic.upper()]
        retrieval_query_text = f"Information about module {question_topic} for exam questions."
    # ... [diğer if/elif/else blokları] ...

    available_topics_for_selection = []
    if target_module_ids:
        for mod_id in target_module_ids:
            if mod_id in MODULE_TOPICS:
                available_topics_for_selection.extend(MODULE_TOPICS[mod_id])
    # ... [diğer topic seçimi kodları] ...

    if not available_topics_for_selection:
        raise HTTPException(status_code=404, detail=f"'{question_topic}' ile ilişkili konu bulunamadı.")

    topics_for_this_batch = random.sample(available_topics_for_selection, min(number_of_questions, len(available_topics_for_selection)))
    if number_of_questions > len(topics_for_this_batch):
        # Allow repetition if not enough unique topics are available
        topics_for_this_batch.extend(random.choices(available_topics_for_selection, k=number_of_questions - len(topics_for_this_batch)))

    all_retrieved_chunks_data = await _retrieve_relevant_chunks(
        retrieval_query_text, 
        module_ids=target_module_ids, 
        file_names=target_file_names, 
        top_k=100
    )

    if not all_retrieved_chunks_data:
        raise HTTPException(status_code=404, detail="Bilgi kaynağında ilgili metin bulunamadı.")

    # --- 2. Adım: Tek ve Toplu API Çağrısı ---

    retrieval_content = ""
    for chunk_data in all_retrieved_chunks_data: 
        retrieval_content += f"--- Kaynak: {chunk_data.get('file_name', 'Bilinmiyor')} ---\n{chunk_data['content']}\n\n"

    existing_questions_prompt_part = ""
    if existing_questions:
        existing_questions_str = json.dumps(existing_questions, ensure_ascii=False)
        existing_questions_prompt_part = f"\nDAHA ÖNCE ÜRETİLMİŞ SORULAR (Bunlardan FARKLI sorular üretmelisin):\n{existing_questions_str}"

    topics_list_str = "\n".join([f"- {topic}" for topic in topics_for_this_batch])

    system_prompt = f"""
GÖREV:
Sen, sağlanan bilgi kaynağına (`knowledge_base`) dayanarak, sana verilen konu listesindeki her bir başlık için BİR TANE olmak üzere, yüksek kaliteli ve birbirinden tamamen farklı çoktan seçmeli sınav soruları üreten bir yapay zekasın.

🎯 AMAÇ:
1.  Sana verilen konu listesindeki (`topics_to_cover`) her bir başlık için, o başlıkla ilgili, bilgi kaynağından bir çoktan seçmeli soru üret.
2.  Her soru için {number_of_choices} adet seçenek üret.

🧷 KURALLAR (Kritik):
1.  **KONUYA UYUM (EN ÖNEMLİ KURAL):** `topics_to_cover` listesindeki her bir başlık için **tam olarak bir adet** soru üretmelisin. Toplamda {number_of_questions} soru üretmiş olmalısın.
2.  **DOĞRU CEVAP KONUMU:** `options` listesindeki her bir iç listede, doğru cevap **her zaman ilk sırada (indeks 0)** olmalıdır. Diğer tüm şıklar mantıklı ama yanlış çeldiriciler olmalıdır.
3.  **KAVRAMSAL BAĞIMSIZLIK:** Üretilen her soru farklı bir fikir veya süreç üzerine olmalıdır. Daha önceki hiçbir soruyla anlamsal olarak %90'dan fazla benzerlik gösteren veya aynı spesifik detayları hedef alan YENİ bir soru üretmek KESİNLİKLE YASAKTIR. Tamamen farklı açılardan, farklı alt konulardan veya farklı detayları sorgulayan özgün sorular oluştur. Bu kurala uyulmaması, görevin tamamen başarısız olduğu anlamına gelir.
{existing_questions_prompt_part}
4.  **SADECE KAYNAK BİLGİSİ:** Yalnızca sağlanan `knowledge_base` metnini kullan.
5.  **ÇIKTI FORMATI:** Çıktın, `{number_of_questions}` elemanlı bir `questions` listesi ve `{number_of_questions}` elemanlı bir iç içe `options` listesi içeren **tek bir JSON nesnesi** olmalıdır.

Bilgi Kaynağı (`knowledge_base`):
{retrieval_content}
"""

    user_prompt = f"""
Aşağıdaki konu başlıklarının her biri için birer tane olmak üzere, toplamda {number_of_questions} adet çoktan seçmeli soru ve her biri için {number_of_choices} şık üret:

Konu Listesi (`topics_to_cover`):
{topics_list_str}
"""
    
    try:
        response_text = await _call_openai_chat_model(system_prompt, user_prompt)
        parsed_response = json.loads(response_text)

        if not (parsed_response.get("questions") and parsed_response.get("options")):
            raise ValueError("Modelden beklenen formatta veri alınamadı.")
        
        generated_questions = parsed_response["questions"]
        generated_choices = parsed_response["options"]

        if len(generated_questions) != number_of_questions:
            print(f"UYARI: Model beklenen sayıda soru üretmedi. Beklenen: {number_of_questions}, Üretilen: {len(generated_questions)}")
            # Handle mismatch if necessary

        # --- 3. Adım: Şıkları Karıştırma ve Doğru Cevap Harfini Belirleme ---
        final_choices = []
        final_correct_answers_letter = []

        for choices_list in generated_choices:
            if not choices_list:
                continue

            correct_answer_text = choices_list[0]
            random.shuffle(choices_list)
            
            correct_answer_index = choices_list.index(correct_answer_text)
            correct_answer_letter = chr(ord('A') + correct_answer_index)
            final_correct_answers_letter.append(correct_answer_letter)

            lettered_choices = [f"{chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices_list)]
            final_choices.append(lettered_choices)

        return {
            "questions": generated_questions,
            "choices": final_choices,
            "correct_answers": final_correct_answers_letter
        }

    except Exception as e:
        print(f"Toplu çoktan seçmeli soru üretiminde hata oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Toplu çoktan seçmeli soru üretimi sırasında bir hata oluştu: {e}")




async def generate_verbal_questions(
    number_of_questions: int, 
    question_topic: str, 
    existing_questions: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    
    # Retrieval için kullanılacak sorgu metni ve filtreleri belirle
    retrieval_query_text = question_topic 
    target_module_ids = [] 
    target_file_names = [] 

    # question_topic'i yorumla... (Bu bölüm generate_open_ended ile aynı)
    if question_topic.upper() in MODULE_FILES:
        target_module_ids = [question_topic.upper()]
        retrieval_query_text = f"Information about the content and key concepts of module {question_topic} for verbal exam questions."
    elif question_topic.upper() in FILE_LOOKUP_MAP:
        target_file_names = [FILE_LOOKUP_MAP[question_topic.upper()]]
        target_module_ids = [FILE_TO_MODULE_MAP[question_topic.upper()]] 
        retrieval_query_text = f"Information for verbal exam questions from file {question_topic}."
    elif ',' in question_topic:
        parts = [m.strip().upper() for m in question_topic.split(',')]
        for part in parts:
            if part not in MODULE_FILES:
                raise HTTPException(status_code=400, detail=f"Geçersiz modül ID'si: '{part}'.")
            target_module_ids.append(part)
        retrieval_query_text = f"Information for verbal exam questions from modules {question_topic}."
    else:
        print(f"DEBUG: '{question_topic}' anahtar kelime olarak yorumlanıyor...")
        found_files_by_keyword = await _find_relevant_files_by_keyword(question_topic, top_n_files=5)
        
        if found_files_by_keyword:
            target_file_names = found_files_by_keyword
            for fname_original in found_files_by_keyword:
                mod_id = FILE_TO_MODULE_MAP.get(fname_original.upper())
                if mod_id and mod_id not in target_module_ids:
                    target_module_ids.append(mod_id)
            retrieval_query_text = f"Information for verbal exam questions related to '{question_topic}' from files: {', '.join(found_files_by_keyword)}."
        else:
            raise HTTPException(status_code=400, detail=f"Geçersiz konu/modül/dosya: '{question_topic}'.")
    
    available_topics_for_selection = []
    if target_module_ids: 
        for mod_id in target_module_ids:
            if mod_id in MODULE_TOPICS:
                available_topics_for_selection.extend(MODULE_TOPICS[mod_id])
    elif target_file_names:
        for fname_original in target_file_names:
            mod_id = FILE_TO_MODULE_MAP.get(fname_original.upper())
            if mod_id and mod_id in MODULE_TOPICS:
                available_topics_for_selection.extend(MODULE_TOPICS[mod_id])
    
    if not available_topics_for_selection:
        raise HTTPException(status_code=404, detail=f"'{question_topic}' ile ilişkili konu bulunamadı.")

    if number_of_questions > len(available_topics_for_selection):
        topics_for_this_batch = random.sample(available_topics_for_selection, len(available_topics_for_selection)) * (number_of_questions // len(available_topics_for_selection))
        topics_for_this_batch.extend(random.sample(available_topics_for_selection, number_of_questions % len(available_topics_for_selection)))
        random.shuffle(topics_for_this_batch) 
    else:
        topics_for_this_batch = random.sample(available_topics_for_selection, number_of_questions)
        
    all_retrieved_chunks_data = await _retrieve_relevant_chunks(
        retrieval_query_text, 
        module_ids=target_module_ids, 
        file_names=target_file_names, 
        top_k=50 
    ) 

    if not all_retrieved_chunks_data:
        raise HTTPException(status_code=404, detail="Bilgi kaynağında ilgili metin bulunamadı.")
    
    existing_questions_prompt_part = ""
    if existing_questions:
        existing_questions_str = json.dumps(existing_questions, ensure_ascii=False, indent=2)
        existing_questions_prompt_part = f"\nDAHA ÖNCE ÜRETİLMİŞ SORULAR (Bunlardan FARKLI sorular üretmelisin):\n{existing_questions_str}"

    generated_questions = []
    generated_feedback_guides = []
    
    max_attempts_per_question = 3 
    
    for i, selected_topic_for_this_question in enumerate(topics_for_this_batch):
        attempt = 0
        question_generated_successfully = False
        
        while attempt < max_attempts_per_question and not question_generated_successfully:
            attempt += 1
            
            num_chunks_for_question = min(random.randint(5, 10), len(all_retrieved_chunks_data)) 
            random_chunks_for_this_question = random.sample(all_retrieved_chunks_data, num_chunks_for_question)
            
            retrieval_content_for_this_question = ""
            for chunk_data in random_chunks_for_this_question: 
                retrieval_content_for_this_question += f"--- Kaynak: {chunk_data.get('file_name', 'Bilinmiyor')} ---\n{chunk_data['content']}\n\n"

            verbal_question_prompt = f"""
GÖREV:
Sen, bir denizcilik akademisinde sözlü sınavlar hazırlayan uzman bir eğitmensin. Görevin, bir öğrencinin bilgisini derinlemesine ölçen, 1-2 dakikalık sözel bir cevap gerektiren sorular hazırlamak ve bu soruları değerlendirecek başka bir eğitmen için detaylı bir geri bildirim rehberi (`feedback_guide`) oluşturmaktır.
SORU STİLİ (KRİTİK):
Sorular, basit bir evet/hayır veya tek kelimelik cevapla geçiştirilememelidir. Öğrenciyi bir prosedürü anlatmaya, bir sistemi açıklamaya veya kavramları karşılaştırmaya teşvik etmelidir.
* **Kullanılacak ifadeler:** "Explain...", "Describe the process of...", "Compare and contrast...", "Walk me through the steps for..."
* **Özellikle '{selected_topic_for_this_question}' konusuyla ilgili bir soru üretmelisin.**
{existing_questions_prompt_part}
GERİ BİLDİRİM REHBERİ (`correct_answers`) STİLİ (KRİTİK):
`correct_answers` alanı, bir "ideal cevap" metni DEĞİLDİR. Bu, bir insan eğitmene, öğrencinin cevabını değerlendirirken nelere dikkat etmesi gerektiğini anlatan bir **yol haritasıdır**.
* **İçerik:** Öğrencinin cevabında bahsetmesi beklenen **tüm anahtar kavramları, teknik terimleri, prosedür adımlarını ve kritik güvenlik notlarını** madde madde listele.
* **Format:** Açık ve anlaşılır olması için maddeleme (`-` veya `*`) kullan.
ZORUNLU ÇIKTI FORMATI:
Çıktın, **kesinlikle ve sadece** `questions` ve `correct_answers` anahtarlarını içeren geçerli bir JSON olmalıdır.
Bilgi Kaynağı Metni:
{retrieval_content_for_this_question}
"""
            
            user_message = f"Yukarıdaki talimatlara göre 1 adet sözel soru ve geri bildirim rehberi üret."
            
            try:
                response_text = await _call_openai_nano_model_json(verbal_question_prompt, user_message)
                parsed_response = json.loads(response_text)
                
                if parsed_response.get("questions") and parsed_response.get("correct_answers"):
                    generated_questions.extend(parsed_response["questions"])
                    generated_feedback_guides.extend(parsed_response["correct_answers"])
                    question_generated_successfully = True
                else:
                    print(f"UYARI: Nano model sözel soru üretiminde boş liste döndü. Soru indeksi: {i}, Deneme: {attempt}")
            except Exception as e:
                print(f"UYARI: Nano model ile sözel soru üretiminde hata oluştu. Soru indeksi: {i}, Deneme: {attempt}, Hata: {e}")
        
        if not question_generated_successfully:
            print(f"HATA: Sözel Soru {i+1} için {max_attempts_per_question} denemede başarılı soru üretilemedi.")
            
    if len(generated_questions) != number_of_questions:
        raise HTTPException(status_code=500, detail=f"Beklenen sözel soru sayısı ({number_of_questions}) üretilemedi. Üretilen: {len(generated_questions)}.")

    return {
        "questions": generated_questions,
        "correct_answers": generated_feedback_guides
    }


# --- Cevap Kontrol Fonksiyonunun Güncellenmesi (Doğrudan Prompt Sistemi) ---
import json
from typing import List
from fastapi import HTTPException














async def provide_feedback_on_verbal_answers(
    questions: List[str],
    feedback_guides: List[str], # Bunlar veritabanının 'correct_answers' sütunundan gelecek
    student_verbal_answers: List[str]
) -> List[str]:

    final_feedbacks = []
    
    tasks = []
    for i in range(len(questions)):
        feedback_provider_prompt = f"""
GÖREV VE KİŞİLİK:
Sen, bir denizcilik akademisinde eğitmen asistanı olan, yardımsever bir yapay zekasın. Görevin, bir öğrencinin sözlü sınav cevabını, sana verilen geri bildirim rehberine (`feedback_guide`) göre analiz etmek ve bu analizi, nihai değerlendirmeyi yapacak olan insan eğitmene sunulmak üzere yapıcı bir geri bildirim metni olarak özetlemektir.

ZORUNLU ÇIKTI FORMATI (KRİTİK):
Çıktın, **kısa, net ve konunun özüne odaklanmalıdır.** Cevabın **kesinlikle yeni satır karakteri (`\\n`) içermemelidir.** Cevabı, aşağıdaki başlıkları kullanarak tek bir paragraf halinde yaz. **Her bölüm için en fazla 1-2 madde belirt.**

GİRDİLER:
* `question`: Öğrenciye sorulan soru.
* `student_answer`: Öğrencinin (sesinden yazıya dökülmüş) cevabı.
* `feedback_guide`: Öğrencinin cevabında olması beklenen anahtar noktaları listeleyen rehber.

SENİN GÖREVİN:
Öğrencinin cevabını rehberle karşılaştır. **Asla `correct` veya `wrong` gibi bir yargıda bulunma.** Sadece objektif bir analiz sun.

ZORUNLU ÇIKTI FORMATI (KRİTİK):
Çıktın, **kısa, net ve konunun özüne odaklanmalıdır.** Cevabın **kesinlikle yeni satır karakteri (`\\n`) içermemelidir.** Bunun yerine, her bölümü başlıklarla ayırarak tek bir paragraf halinde yaz.

FORMAT ŞÖYLE OLMALIDIR:
**Güçlü Yönler:** (Öğrencinin, rehberdeki hangi noktalara doğru bir şekilde değindiğini 1-2 cümleyle özetle.) **Geliştirilebilecek Yönler:** (Öğrencinin, rehberdeki hangi önemli noktaları atladığını veya yanlış açıkladığını 1-2 cümleyle özetle.) **Genel Özet:** (Öğrencinin konuyu anlama seviyesi hakkında tek bir cümlelik genel bir yorum yap.)
"""
        
        user_message = f"""
        Aşağıdaki verileri kullanarak geri bildirim metnini oluştur:

        Soru: {questions[i]}
        
        Geri Bildirim Rehberi (Beklenenler):
        {feedback_guides[i]}

        Öğrencinin Cevabı:
        {student_verbal_answers[i]}
        """
        tasks.append(_call_openai_nano_model_text(feedback_provider_prompt, user_message))

    try:
        # Tüm geri bildirim görevlerini eş zamanlı olarak çalıştır
        final_feedbacks = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Geri bildirim üretilirken kritik bir asyncio hatası oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Geri bildirim üretilirken bir hata oluştu: {e}")

    return final_feedbacks



# --- Veritabanı İşlemleri (exam_name EKLENEREK GÜNCELLENDİ) ---

async def get_student_exam_record(exam_name: str, student_name: str, question_type: str) -> dict | None:
    """Belirtilen sınav adı, öğrenci ve soru tipi için tek bir sınav kaydını getirir."""
    try:
        response = await run_in_threadpool(
            lambda: supabase.table('exam_records')
            .select("*")
            .eq("exam_name", exam_name)
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
    """Bir sınav kaydını ekler veya günceller. Çakışma durumu (exam_name, student_name, question_type) ile kontrol edilir."""
    try:
        if not all(k in record_data for k in ['exam_name', 'student_name', 'question_type']):
            raise ValueError("upsert_exam_record için exam_name, student_name ve question_type zorunludur.")
            
        response = await run_in_threadpool(
            lambda: supabase.table('exam_records')
            .upsert(record_data, on_conflict='exam_name,student_name,question_type')
            .execute()
        )
        return response.data[0]
    except Exception as e:
        print(f"Sınav kaydı eklenirken/güncellenirken hata oluştu: {e}")
        return None

async def update_all_questions_in_record(exam_name: str, student_name: str, question_type: str, new_questions: List[str], new_correct_answers: Optional[List[str]] = None) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        record = {"exam_name": exam_name, "student_name": student_name, "question_type": question_type}

    record['questions'] = new_questions
    
    if new_correct_answers is not None:
        if len(new_questions) != len(new_correct_answers):
            raise ValueError("Soru sayısı ile doğru cevap sayısı eşleşmelidir.")
        record['correct_answers'] = new_correct_answers
    
    return await upsert_exam_record(record)

async def update_all_choices_in_record(exam_name: str, student_name: str, question_type: str, all_new_choices: List[List[str]]) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        record = {"exam_name": exam_name, "student_name": student_name, "question_type": question_type}

    record['choices'] = all_new_choices
    
    return await upsert_exam_record(record)

async def update_answer(exam_name: str, student_name: str, question_type: str, index: int, answer: str) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    
    if not record:
        record = {"exam_name": exam_name, "student_name": student_name, "question_type": question_type, "answers": []}

    answers = record.get('answers')
    if not isinstance(answers, list):
        answers = []
        record['answers'] = answers

    while len(answers) <= index:
        answers.append(None)

    answers[index] = answer
    
    record['answers'] = answers
    
    return await upsert_exam_record(record)

async def update_answers_bulk(exam_name: str, student_name: str, question_type: str, new_answers: List[str]) -> dict | None:
    return await upsert_exam_record({
        "exam_name": exam_name,
        "student_name": student_name,
        "question_type": question_type,
        "answers": new_answers
    })

async def update_results_bulk(exam_name: str, student_name: str, question_type: str, new_results: List[str]) -> dict | None:
    return await upsert_exam_record({
        "exam_name": exam_name,
        "student_name": student_name,
        "question_type": question_type,
        "results": new_results
    })

async def update_plagiarism_violations_in_record(exam_name: str, student_name: str, question_type: str, violation_text: str) -> dict | None:
    return await upsert_exam_record({
        "exam_name": exam_name,
        "student_name": student_name,
        "question_type": question_type,
        "plagiarism_violations": violation_text
    })

async def get_questions_all(exam_name: str, student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('questions') if record and isinstance(record.get('questions'), list) else None

async def get_answers_all(exam_name: str, student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('answers') if record and isinstance(record.get('answers'), list) else None

async def get_results_all(exam_name: str, student_name: str, question_type: str) -> List[str] | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('results') if record and isinstance(record.get('results'), list) else None

async def get_total_score(exam_name: str, student_name: str, question_type: str) -> float | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('total_score') if record and 'total_score' in record else None

async def get_plagiarism_violations(exam_name: str, student_name: str, question_type: str) -> str | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('plagiarism_violations') if record and 'plagiarism_violations' in record else None


async def update_all_correct_answers_in_record(exam_name: str, student_name: str, question_type: str, new_correct_answers: List[str]) -> dict | None:
    """Bir sınavdaki tüm doğru cevapları toplu olarak günceller."""
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        record = {"exam_name": exam_name, "student_name": student_name, "question_type": question_type}
    
    record['correct_answers'] = new_correct_answers
    return await upsert_exam_record(record)

async def get_correct_answers_all(exam_name: str, student_name: str, question_type: str) -> List[str] | None:
    """Bir sınavdaki tüm doğru cevapları döndürür."""
    record = await get_student_exam_record(exam_name, student_name, question_type)
    return record.get('correct_answers') if record and isinstance(record.get('correct_answers'), list) else None

async def delete_single_question(exam_name: str, student_name: str, question_type: str, index: int) -> dict | None:
    """Belirtilen indeksteki bir soruyu ve ilgili tüm verilerini siler."""
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        return None

    questions = record.get('questions')
    
    if not questions:
        raise ValueError("Silinecek soru bulunmuyor (soru listesi boş).")
        
    if not (0 <= index < len(questions)):
        raise ValueError(f"Geçersiz indeks: {index}. Soru sayısı: {len(questions)}.")

    for key in ['questions', 'correct_answers', 'answers', 'results', 'choices']:
        if key in record and isinstance(record.get(key), list) and index < len(record[key]):
            record[key].pop(index)

    return await upsert_exam_record(record)


async def delete_all_questions(exam_name: str, student_name: str, question_type: str) -> dict | None:
    """Belirtilen sınav türündeki tüm soruları ve ilgili verileri null olarak ayarlar."""
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        return None

    if not record.get('questions'):
        raise ValueError("Silinecek soru bulunmuyor (soru listesi zaten boş).")

    record['questions'] = None
    record['correct_answers'] = None
    record['answers'] = None
    record['results'] = None
    record['choices'] = None
    record['total_score'] = None

    return await upsert_exam_record(record)

# main.py'deki çağırma uyumluluğunu sağlamak için eklenen/güncellenen fonksiyonlar
async def get_question(exam_name: str, student_name: str, question_type: str, index: int) -> str | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if record and isinstance(record.get('questions'), list) and index < len(record['questions']):
        return record['questions'][index]
    return None

async def get_choice(exam_name: str, student_name: str, question_type: str, question_index: int, choice_index: int) -> str | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if record and isinstance(record.get('choices'), list) and question_index < len(record['choices']):
        if isinstance(record['choices'][question_index], list) and choice_index < len(record['choices'][question_index]):
            return record['choices'][question_index][choice_index]
    return None

async def update_question_in_record(exam_name: str, student_name: str, question_type: str, question_index: int, value: str, correct_answer: Optional[str] = None) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record:
        return None

    questions = record.get('questions', [])
    if not isinstance(questions, list):
        questions = []
    while len(questions) <= question_index:
        questions.append(None)
    questions[question_index] = value
    record['questions'] = questions

    if correct_answer is not None:
        correct_answers = record.get('correct_answers', [])
        if not isinstance(correct_answers, list):
            correct_answers = []
        while len(correct_answers) <= question_index:
            correct_answers.append(None)
        correct_answers[question_index] = correct_answer
        record['correct_answers'] = correct_answers

    return await upsert_exam_record(record)

async def update_choice_in_record(exam_name: str, student_name: str, question_type: str, question_index: int, choice_index: int, value: str) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record or not isinstance(record.get('choices'), list):
        return None

    choices = record['choices']
    if not (isinstance(choices, list) and question_index < len(choices) and isinstance(choices[question_index], list) and choice_index < len(choices[question_index])):
        return None
    
    choices[question_index][choice_index] = value
    record['choices'] = choices

    return await upsert_exam_record(record)

async def update_choices_for_single_question_in_record(exam_name: str, student_name: str, question_type: str, question_index: int, new_choices: List[str]) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record or not isinstance(record.get('choices'), list):
        return None

    choices = record.get('choices', [])
    while len(choices) <= question_index:
        choices.append([])
    choices[question_index] = new_choices
    record['choices'] = choices
    
    return await upsert_exam_record(record)

async def update_correct_answer_in_record(exam_name: str, student_name: str, question_type: str, index: int, correct_answer: str) -> dict | None:
    """Belirli bir sorunun doğru cevabını günceller."""
    record = await get_student_exam_record(exam_name, student_name, question_type)
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

async def update_result(exam_name: str, student_name: str, question_type: str, index: int, result: str) -> dict | None:
    record = await get_student_exam_record(exam_name, student_name, question_type)
    if not record or not isinstance(record.get('results'), list):
        if not record:
            record = {"exam_name": exam_name, "student_name": student_name, "question_type": question_type, "results": []}
        else:
            record['results'] = []
    
    results = record['results']
    while len(results) <= index:
        results.append(None)

    results[index] = result
    record['results'] = results
    return await upsert_exam_record(record)


from openai import AsyncOpenAI # Make sure AsyncOpenAI is imported
# new import for file handling
import io 

# existing client setup
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY) # Make sure this line exists and is correct

# --- NEW: Function to handle voice answers ---

async def add_voice_answer(
    exam_name: str,
    student_name: str,
    index: int,
    audio_file: io.BytesIO # io.BytesIO tipinde bir dosya objesi bekliyoruz
) -> str:
    """
    Verilen ses dosyasını OpenAI Whisper kullanarak metne çevirir ve
    belirli bir sınav kaydındaki cevabı günceller.
    question_type'ın 'Verbal Question' olması beklenir.
    """
    # 1. Ses dosyasını metne çevirmek için Whisper API'ını kullan
    # In examai.py, inside the add_voice_answer function:

    # In examai.py, inside the add_voice_answer function:

    try:
        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        transcribed_text = transcription.strip() # CORRECTED LINE: Directly use 'transcription' as it's already the string
        if not transcribed_text:
            raise ValueError("Ses metne çevrilemedi veya boş bir metin döndürüldü.")

    except Exception as e:
        print(f"Whisper API hatası: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ses metne çevrilirken hata oluştu: {e}")
    
    # 2. Metne çevrilen cevabı veri tabanına kaydet
    try:
        # update_answer fonksiyonunu kullanarak cevabı güncelliyoruz
        # question_type'ı "Verbal Question" olarak sabitliyoruz.
        updated_record = await update_answer(
            exam_name=exam_name,
            student_name=student_name,
            question_type="Verbal Question", # Bu kısım sabit!
            index=index,
            answer=transcribed_text
        )

        if not updated_record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sınav kaydı bulunamadığı için sesli cevap kaydedilemedi.")
            
        return transcribed_text

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Veri tabanına sesli cevap kaydedilirken hata: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Sesli cevap veri tabanına kaydedilirken hata oluştu: {e}")
    



async def check_answers_in_batch_with_rubrics(
    questions_with_topics: List[Dict[str, str]],
    evaluation_rubrics: List[Dict],
    answers: List[str],
    batch_size: int = 10
) -> Dict[str, List[str]]:
    """
    Cevapları, her soru için özel olarak üretilmiş ve VE/VEYA mantığı içerebilen
    bir Rubric kullanarak toplu halde değerlendirir. Bu versiyon, en gelişmiş
    değerlendirme mantığını kullanır.
    """
    
    all_data = list(zip(questions_with_topics, evaluation_rubrics, answers))
    batches = [all_data[i:i + batch_size] for i in range(0, len(all_data), batch_size)]
    
    print(f"\n--- Rubric ile Toplu Değerlendirme Başladı: {len(questions_with_topics)} soru, {len(batches)} parça... ---")

    tasks = []
    
    # --- DEĞİŞİKLİK BURADA BAŞLIYOR: ESKİ PROMPT, NİHAİ PROMPT İLE DEĞİŞTİRİLDİ ---
    
    auditor_system_prompt = """
GÖREV VE KİŞİLİK:
Sen, adı 'AuditorAI' olan, duygusal olmayan, son derece tutarlı ve sadece sağlanan kanıtlara dayanan bir denetim yapay zekasısın. Görevin, bir öğrenci cevabını, sana verilen spesifik Değerlendirme Kriterleri'ne (Rubric) göre analiz ederek cevabın yeterliliğini ve bütünlüğünü değerlendirmektir.

GİRDİLER:
Sana her zaman üç anahtar bilgi verilecek:
1.  question: İçinde hem `topic` (konu başlığı) hem de `question` (soru metni) bulunan bir JSON nesnesi.
2.  student_answer: Öğrencinin verdiği cevap.
3.  evaluation_rubric: Cevabı değerlendirmek için kullanacağın kriterler (`anahtar_kavram`, `kabul_kriterleri`, `ret_kriterleri`).

ZORUNLU KARAR VERME ALGORİTMASI (TARTIŞMASIZ):
Her bir öğrenci cevabını değerlendirirken aşağıdaki adımları kesinlikle bu sırayla izleyeceksin:

Adım 1: Ret Kriterlerini Kontrol Et (Kesin Ret Adımı)
Öğrencinin cevabını dikkatlice oku ve `ret_kriterleri` listesindeki maddelerden herhangi birini tetikleyip tetiklemediğini kontrol et.
* Eğer cevap, `ret_kriterleri` listesindeki maddelerden BİRİNİ BİLE tetikliyorsa, düşünmeyi anında durdur. Karar `wrong` olmalıdır. Gerekçe olarak tetiklenen ret kriterini belirt.

Adım 2: Kabul Kriterlerini Ayrıştırma ve Doğrulama (Mantıksal Kontrol Adımı)
Eğer cevap Adım 1'i geçtiyse, şimdi `kabul_kriterleri` listesindeki her bir maddeyi bir mantıksal kural olarak ele al ve doğrula:

* Adım 2a: Mantıksal Kuralları Yorumlama
    * VE (AND) Kuralı: Eğer bir kabul kriteri içinde `VE` operatörü varsa, bu, o kuralın geçerli sayılması için TÜM koşulların cevapta bulunmasının ZORUNLU olduğu anlamına gelir. Bir tanesi bile eksikse, o kural karşılanmamış sayılır.
    * VEYA (OR) Kuralı: Eğer bir kabul kriteri içinde `VEYA` operatörü varsa, bu, koşullardan en az birinin cevapta bulunmasının YETERLİ olduğu anlamına gelir.
    * Basit Kural: Eğer bir kriterde `VE`/`VEYA` yoksa, o tek bir koşul olarak değerlendirilir.

* Adım 2b: Karar Verme
    * `correct` Kararı İçin: Cevabın `correct` sayılabilmesi için, `kabul_kriterleri` listesindeki en az bir ana mantıksal kuralı tam olarak (yani içindeki tüm `VE` koşullarıyla birlikte) karşılaması gerekir.
    * `wrong` Kararı İçin: Eğer cevap, `kabul_kriterleri` listesindeki hiçbir ana mantıksal kuralı tam olarak karşılamıyorsa, karar `wrong` olmalıdır. Bu durum, cevabın hem tamamen yetersiz olmasını hem de "Eksik Bilgi Kör Noktası"nı (yani bir `VE` kuralının sadece bir kısmını karşılamasını) kapsar.

GEREKÇE YAZMA KURALLARI (ZORUNLU):
Gerekçen, yukarıdaki algoritmayı nasıl uyguladığını yansıtmalıdır.

* `correct` ise: Gerekçen, öğrencinin karşıladığı temel kabul kriterini veya kuralını kısaca özetlemelidir.
    * Örnek: "Doğru: Cevap, 'tüm vinç tiplerini listeleme' kuralını karşılamakta ve 'single drum', 'twin drum', 'electric' ve 'hydraulic' ifadelerini içermektedir."

* `wrong` ise: Gerekçen, kararın nedenini (Adım 1 mi, Adım 2 mi) net bir şekilde belirtmelidir.
    * Ret Kriteri Tetiklendiyse (Adım 1): "Yanlış: Cevap, 'jet propulsion sistemi için propeller kelimesi' ret kriterini doğrudan tetiklemiştir."
    * Mantıksal Kural Karşılanmadıysa (Adım 2): "Yanlış: Cevap yetersiz çünkü 'tüm can salı tiplerini listeleme' kuralını karşılamamaktadır; zorunlu olan 'freefall lifeboat' ifadesi eksiktir."
    * Hiçbir Kural Karşılanmadıysa (Adım 2): "Yanlış: Cevap, beklenen kabul kriterlerinden hiçbirini (örneğin, vinç tipleri veya güç kaynakları) içermediği için konu dışı veya tamamen yetersizdir."

ZORUNLU ÇIKTI FORMATI:
Çıktın, kesinlikle ve sadece `{"results": [...], "reasonings": [...]}` formatında geçerli bir JSON olmalıdır.
"""
    # --- DEĞİŞİKLİK BURADA BİTİYOR ---

    for i, batch_data in enumerate(batches):
        print(f"--- Parça {i+1}/{len(batches)} hazırlanıyor... ---")
        
        batch_questions = [item[0] for item in batch_data]
        batch_rubrics = [item[1] for item in batch_data]
        batch_answers = [item[2] for item in batch_data]
        
        input_data = {
            "questions": batch_questions,
            "evaluation_rubrics": batch_rubrics,
            "student_answers": batch_answers
        }
        
        user_message_content = json.dumps(input_data, ensure_ascii=False, indent=2)
        tasks.append(_call_openai_chat_model(auditor_system_prompt, user_message_content))

    final_results = []
    final_reasonings = []

    try:
        assistant_batch_responses = await asyncio.gather(*tasks)

        for i, response_text in enumerate(assistant_batch_responses):
            try:
                cleaned_text = response_text.strip().strip('`').strip('json\n').strip()
                parsed_response = json.loads(cleaned_text)
                
                if (isinstance(parsed_response, dict) and 
                        "results" in parsed_response and 
                        "reasonings" in parsed_response):
                    
                    batch_results = parsed_response["results"]
                    batch_reasonings = parsed_response["reasonings"]
                    
                    if len(batch_results) != len(batches[i]) or len(batch_reasonings) != len(batches[i]):
                         print(f"UYARI: Parça {i+1} için Asistan'dan beklenmedik sayıda sonuç/gerekçe döndü.")
                         final_results.extend(["wrong"]* len(batches[i]))
                         final_reasonings.extend(["Hatalı batch boyutu nedeniyle geçersiz sayıldı."] * len(batches[i]))
                    else:
                        print(f"--- Parça {i+1} Başarıyla Değerlendirildi. ---")
                        for j, reasoning in enumerate(batch_reasonings):
                            print(f"  -> Karar='{batch_results[j]}', Gerekçe='{reasoning}'")
                        
                        final_results.extend(batch_results)
                        final_reasonings.extend(batch_reasonings)
                else:
                    print(f"UYARI: Parça {i+1} için Asistan'dan beklenmedik formatta yanıt.")
                    final_results.extend(["wrong"]* len(batches[i]))
                    final_reasonings.extend([f"Beklenmedik format: {response_text}"] * len(batches[i]))

            except json.JSONDecodeError:
                print(f"UYARI: Parça {i+1} için Asistan'dan JSON olmayan yanıt.")
                final_results.extend(["wrong"] * len(batches[i]))
                final_reasonings.extend([f"JSON olmayan yanıt: {response_text}"] * len(batches[i]))
    
    except Exception as e:
        print(f"Cevap kontrolü sırasında kritik bir asyncio hatası oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Asistan görevleri çalıştırılırken bir hata oluştu: {e}")

    if len(final_results) != len(questions_with_topics):
        raise HTTPException(status_code=500, detail="Değerlendirme sonrası toplam sonuç sayısı, soru sayısıyla eşleşmiyor.")

    print("--- Toplu Değerlendirme Tamamlandı. ---")
    
    return {
        "results": final_results,
        "reasonings": final_reasonings
    }
