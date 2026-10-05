import os
import itertools
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KEYS_ENV = os.getenv("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]
key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

class SolveRequest(BaseModel):
    stage: str
    grade: str
    subject: str
    question: Optional[str] = ""
    image_base64: Optional[str] = None

@app.get("/")
def health_check():
    return {"status": "running", "active_keys": len(API_KEYS)}

@app.post("/solve")
async def solve_question(req: SolveRequest):
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="Server keys not configured")

    current_key = next(key_cycle)
    genai.configure(api_key=current_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = f"""
أنت عضو اللجنة الدائمة للامتحانات والمصحح الأول في مركز الفحص والتقويم بوزارة التربية العراقية.
المرحلة الدراسية: {req.stage} | الصف: {req.grade} | المادة: {req.subject}
السؤال: {req.question if req.question else "حل السؤال الموضح في الصورة بدقة وتفصيل"}

تعليمات التصحيح الوزاري الصارمة:
1. التزم حصراً بالأجوبة النموذجية المعتمدة لوزارة التربية العراقية ومركز الفحص.
2. قواعد اللغة العربية: الإعراب المفصل، القاعدة المقررة، وسبب الإعراب وفق مصطلحات المنهج العراقي.
3. المسائل العلمية (رياضيات/فيزياء/كيمياء): [المعطيات]، [القانون الرسمي]، [التعويض الرياضي]، [الناتج النهائي مع الوحدات].
4. المواد الشرحية والحفظية: نص الكتاب الرسمي دون تلخيص تعبيري شخصي.
5. اذكر في البداية اسم الفصل أو الموضوع المنهجي.
"""

    contents = [system_prompt]
    if req.image_base64:
        contents.append({
            "mime_type": "image/jpeg",
            "data": req.image_base64
        })

    try:
        response = model.generate_content(contents)
        return {"solution": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="خطأ أثناء معالجة السؤال، يرجى المحاولة مجدداً.")
