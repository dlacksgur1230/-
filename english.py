import re
import json
import random
import ssl
import base64
import urllib.request
import urllib.parse
from difflib import SequenceMatcher
import streamlit as st
import streamlit.components.v1 as components

# SSL 인증서 검증 우회
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------------------------------------
# 1. 페이지 설정 및 디자인(CSS)
# ---------------------------------------------------------
st.set_page_config(page_title="영어 지문 분석기 (Aikar AI 완벽판)", layout="wide")

CUSTOM_CSS = """
<style>
.circle-word { border: 1.5px solid #4e73df; border-radius: 12px; padding: 3px 9px; margin: 3px 2px; display: inline-block; background-color: #ffffff; font-size: 16px; color: #2e59d9; font-weight: 600; }
.slash { color: #e74a3b; font-weight: bold; font-size: 20px; margin: 0 8px; }
.sentence-container { background-color: #f8f9fc; padding: 15px; border-radius: 8px; border-left: 5px solid #4e73df; margin-bottom: 8px; line-height: 2.3; }
.question-box { background-color: #ffffff; border: 1px solid #e3e6f0; border-left: 4px solid #1cc88a; padding: 15px; border-radius: 6px; margin-bottom: 12px; line-height: 1.8; }
.summary-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
.score-box { background-color: #e8f4f8; border: 2px solid #36b9cc; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center; font-size: 18px; font-weight: bold; color: #2c3e50; }
.grammar-error { border: 2px solid #e74a3b !important; background-color: #fdf2f2 !important; color: #e74a3b !important; }
.grammar-error u { text-decoration: underline red 3px; font-weight: bold; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 세션 상태 관리
# ---------------------------------------------------------
if "sentences" not in st.session_state: st.session_state.sentences = []
if "translations" not in st.session_state: st.session_state.translations = []
if "mode" not in st.session_state: st.session_state.mode = None
if "generated_questions" not in st.session_state: st.session_state.generated_questions = []
if "summary_data" not in st.session_state: st.session_state.summary_data = None
if "eng_input_area" not in st.session_state: st.session_state.eng_input_area = ""

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we", "us", "our", "you", "your"}

# ---------------------------------------------------------
# 3. AI 연동 및 Fallback 파이프라인 (8000 토큰, 120초 제한)
# ---------------------------------------------------------
AIKAR_API_URL = "https://chat.aeonthic.com/aikar-engine/v1/chat/completions"
OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"

def call_aikar_ai(prompt, system_prompt, timeout=120):
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    payload = {
        "model": "Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "max_tokens": 8000
    }
    try:
        req = urllib.request.Request(AIKAR_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res = json.loads(response.read().decode("utf-8"))
            if "choices" in res and len(res["choices"]) > 0: return res["choices"][0]["message"]["content"]
    except Exception: pass
    return None

def call_ollama_ai(prompt, system_prompt, timeout=60):
    headers = {"Content-Type": "application/json"}
    payload = {"model": "llama3.2", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        req = urllib.request.Request(OLLAMA_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res = json.loads(response.read().decode("utf-8"))
            if "choices" in res and len(res["choices"]) > 0: return res["choices"][0]["message"]["content"]
    except Exception: pass
    return None

def call_ai_with_fallback(prompt, system_prompt):
    res = call_aikar_ai(prompt, system_prompt, timeout=120)
    if res: return res
    
    res_ollama = call_ollama_ai(prompt, system_prompt, timeout=60)
    if res_ollama:
        st.toast("🌐 외부 AI 미응답으로 인해 [로컬 Ollama AI]로 자동 전환되었습니다.", icon="💻")
        return res_ollama
        
    st.toast("⚠️ AI 미연결 상태입니다. 오프라인 내장 알고리즘으로 전환합니다.", icon="🔌")
    return None

def extract_text_from_image_via_ai(image_file):
    try:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_image}"
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        payload = {
            "model": "Lumen-3.5-Pulsar_S-LD-Q4_0_XL.gguf",
            "messages": [
                {"role": "system", "content": "Extract all English text accurately. Return ONLY the text."},
                {"role": "user", "content": [{"type": "text", "text": "Extract text."}, {"type": "image_url", "image_url": {"url": data_url}}]}
            ],
            "max_tokens": 8000
        }
        req = urllib.request.Request(AIKAR_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            res = json.loads(response.read().decode("utf-8"))
            if "choices" in res and len(res["choices"]) > 0:
                return re.sub(r'^```[a-zA-Z]*\s*|```\$', '', res["choices"][0]["message"]["content"].strip(), flags=re.MULTILINE).strip(), None
    except Exception: return None, "AI 이미지 서버에 연결할 수 없습니다. 텍스트를 직접 입력해 주세요."
    return None, "텍스트를 추출하지 못했습니다."

# ---------------------------------------------------------
# 4. 분석, 번역, 요약, 문법 로직
# ---------------------------------------------------------
def split_into_sentences(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) == 1:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', lines[0]) if s.strip()]
    return lines

def process_inputs(eng_text, kor_text):
    eng_sentences = split_into_sentences(eng_text)
    kor_sentences = split_into_sentences(kor_text) if kor_text.strip() else []
    
    if not kor_sentences and eng_sentences:
        prompt = f"다음 영어 문장을 한국어로 자연스럽게 번역해 JSON 배열 형태로만 응답해.\n{json.dumps(eng_sentences, ensure_ascii=False)}"
        res = call_ai_with_fallback(prompt, "Return ONLY a JSON array of strings.")
        if res:
            try: kor_sentences = [str(i).strip() for i in json.loads(re.sub(r'^```json\s*|```\$', '', res.strip(), flags=re.MULTILINE))]
            except: kor_sentences = [re.sub(r'^(?:\d+[\.\)]|[-•*])\s*', '', l.strip()) for l in res.splitlines() if l.strip() and not l.startswith("```")]

    translations = [kor_sentences[i] if i < len(kor_sentences) else "(오프라인 상태: 한국어 해석을 직접 입력해 주세요)" for i in range(len(eng_sentences))]
    return eng_sentences, translations

def generate_offline_vocab_and_summary(sentences):
    full_text = " ".join(sentences)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
    
    freq = {}
    for w in words:
        if w not in STOPWORDS: freq[w] = freq.get(w, 0) + 1
            
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    vocab_list = []
    
    for word, count in sorted_words:
        context_sent = next((s for s in sentences if word in s.lower()), "지문 내 수록 어휘")
        vocab_list.append({
            "word": word.capitalize(),
            "meaning": "사전/문맥 직접 확인 필요 (오프라인 모드)",
            "context": f"지문 내 {count}회 등장: '{context_sent[:50]}...'"
        })
        
    topic = f"주요 어휘: {', '.join([w[0].capitalize() for w in sorted_words[:3]])}"
    main_idea = f"이 지문은 총 {len(sentences)}개 문장으로 구성된 내용입니다."
    return {"topic": topic, "main_idea": main_idea, "vocab": vocab_list}

def generate_summary_and_vocab(sentences):
    full_text = " ".join(sentences)
    prompt = f"""다음 지문을 분석해서 JSON 형식으로 출력해줘.
    지문: "{full_text}"
    출력 형식: {{"topic": "핵심주제 1문장", "main_idea": "지문요지 1~2문장", "vocab": [{{"word": "영어단어", "meaning": "한국어 뜻", "context": "문맥상 쓰임"}}]}}"""
    res = call_ai_with_fallback(prompt, "You are a reading comprehension expert. Return ONLY valid JSON.")
    if res:
        try: return json.loads(re.sub(r'^```json\s*|```$', '', res.strip(), flags=re.MULTILINE))
        except: pass
        
    return generate_offline_vocab_and_summary(sentences)

def check_local_rules(sentence):
    rules = [(r'\b(this|that)\s+(are|were)\b', "'this/that' 뒤 단수 동사 필요", "is"), (r'\b(these|those)\s+(is|was)\b', "'these/those' 뒤 복수 동사 필요", "are")]
    return [(m.start(), m.end(), msg, [repl]) for pattern, msg, repl in rules for m in re.finditer(pattern, sentence, re.IGNORECASE)]

def format_sentence_html(sentence, error_spans):
    err_idxs = {i for s, e, _, _ in error_spans for i in range(s, e)}
    html = ""
    curr = 0
    for token in re.split(r'(\s+)', sentence):
        if not token.strip(): html += token; curr += len(token); continue
        is_err = any(i in err_idxs for i in range(curr, curr+len(token)))
        if re.match(r'(\b(?:in|on|at|to|for|with|by|about|because|although|when|where|which|who|that|and|but)\b)', token, re.I):
            html += " <span class='slash'>/</span> "
        html += f"<span class='circle-word grammar-error'><u>{token}</u></span>" if is_err else f"<span class='circle-word'>{token}</span>"
        curr += len(token)
    return html

def render_tts_button(sentence, height=42):
    safe_sent = json.dumps(sentence)
    tts_code = f"""
    <div style="display: flex; align-items: center; margin-bottom: 8px;">
        <button onclick='window.speechSynthesis.cancel(); let msg = new SpeechSynthesisUtterance({safe_sent}); msg.lang="en-US"; window.speechSynthesis.speak(msg);' 
                style="background-color: #4e73df; color: white; border: none; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 13px;">
            🔊 문장 듣기
        </button>
    </div>
    """
    components.html(tts_code, height=height)

# ---------------------------------------------------------
# 5. 수능 5대 원형 문제 한 번에 일괄 생성 (Batch Generation)
# ---------------------------------------------------------
def generate_offline_fallback_questions(sentences):
    full = " ".join(sentences)
    words = [w for w in set(re.findall(r'\b[a-zA-Z]{5,}\b', full)) if w.lower() not in STOPWORDS]
    w1 = words[0] if len(words)>0 else "meaning"
    w2 = words[1] if len(words)>1 else "context"
    
    s_part1 = sentences[0] if len(sentences)>0 else full
    s_part2 = sentences[1] if len(sentences)>1 else full
    s_part3 = sentences[2] if len(sentences)>2 else full

    return [
        {"type": "1. 빈칸 추론", "title": "다음 글의 빈칸에 들어갈 말로 가장 적절한 것은?", "content": full.replace(w1, "[ _______ ]", 1), "options": [f"① {w1}", f"② {w2}", "③ limitation", "④ contradiction", "⑤ illusion"], "answer": f"① {w1}", "explanation": "지문의 핵심 어휘를 빈칸에 대입하는 문제입니다."},
        {"type": "2. 어법 판단", "title": "다음 글의 밑줄 친 부분 중, 어법상 어색한 것은?", "content": f"<b>(A) {s_part1}</b><br><b>(B) {s_part2}</b>", "options": ["① (A)", "② (B)", "③ (C)", "④ (D)", "⑤ (E)"], "answer": "① (A)", "explanation": "문법적 오류를 판별하는 유형입니다."},
        {"type": "3. 어휘 적절성", "title": "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?", "content": full, "options": [f"① {w1}", f"② {w2}", "③ increase", "④ decrease", "⑤ maintain"], "answer": "③ increase", "explanation": "문맥 흐름에 부합하는 반의어 관계를 묻는 문제입니다."},
        {"type": "4. 순서 배열", "title": "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?", "content": f"<b>[제시문] {s_part1}</b><br><br>(A) {s_part2}<br>(B) {s_part3}", "options": ["① (A) - (C) - (B)", "② (B) - (A) - (C)", "③ (B) - (C) - (A)", "④ (C) - (A) - (B)", "⑤ (C) - (B) - (A)"], "answer": "② (B) - (A) - (C)", "explanation": "글의 논리적 선후 관계를 배열하는 문제입니다."},
        {"type": "5. 문장 삽입", "title": "글의 흐름으로 보아, 주어진 문장이 들어 가기에 가장 적절한 곳은?", "content": f"<b>[보기] {s_part1}</b><br><br>{full}", "options": ["① [1]", "② [2]", "③ [3]", "④ [4]", "⑤ [5]"], "answer": "① [1]", "explanation": "문맥상 지시어와 연결어를 고려해 위치를 찾습니다."}
    ]

def generate_5_exam_questions(sentences):
    """단 한 번의 AI 요청으로 5개 유형의 문제를 한 번에 생성하는 로직"""
    full_passage = " ".join(sentences)
    
    prompt = f"""다음 지문을 읽고 5개 수능 핵심 유형의 문제를 한 번에 완전히 작성해줘.

    [지문 내용]
    "{full_passage}"

    [필수 출제 유형 (총 5문제)]
    1. 빈칸 추론 (지문의 핵심 키워드나 주제문 빈칸 처리)
    2. 어법 판단 (밑줄 친 부분 중 어색한 어법 찾기)
    3. 어휘 적절성 (문맥상 어색한 낱말 찾기)
    4. 순서 배열 (제시문 후 이어질 글의 논리적 순서)
    5. 문장 삽입 (주어진 문장이 들어갈 적절한 위치)

    [출력 요구사항]
    반드시 정확히 5개의 문제 객체를 담은 단일 JSON 배열만 반환해. 다른 텍스트나 인사는 포함하지 마.

    [출력 JSON 예시]
    [
      {{"type": "1. 빈칸 추론", "title": "다음 글의 빈칸에 들어갈 말로 가장 적절한 것은?", "content": "지문...", "options": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ..."], "answer": "① ...", "explanation": "해설..."}},
      {{"type": "2. 어법 판단", "title": "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?", "content": "지문...", "options": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ..."], "answer": "② ...", "explanation": "해설..."}},
      {{"type": "3. 어휘 적절성", "title": "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?", "content": "지문...", "options": ["① ...", "② ...", "③ ...", "④ ...", "⑤ ..."], "answer": "③ ...", "explanation": "해설..."}},
      {{"type": "4. 순서 배열", "title": "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?", "content": "지문...", "options": ["① (A)-(C)-(B)", "② (B)-(A)-(C)", "③ (B)-(C)-(A)", "④ (C)-(A)-(B)", "⑤ (C)-(B)-(A)"], "answer": "② (B)-(A)-(C)", "explanation": "해설..."}},
      {{"type": "5. 문장 삽입", "title": "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?", "content": "지문...", "options": ["① [1]", "② [2]", "③ [3]", "④ [4]", "⑤ [5]"], "answer": "① [1]", "explanation": "해설..."}}
    ]"""
    
    res = call_ai_with_fallback(prompt, "You are a CSAT exam creator. Return ONLY a single JSON array containing all 5 questions in one batch.")
    if res:
        try:
            parsed = json.loads(re.sub(r'^```json\s*|```$', '', res.strip(), flags=re.MULTILINE))
            if isinstance(parsed, list) and len(parsed) == 5:
                return parsed
        except: pass
        
    return generate_offline_fallback_questions(sentences)

def create_export_text(questions):
    text = "📝 영어 지문 생성 문제지 (수능 5개 핵심 유형)\n" + "="*40 + "\n\n"
    for i, q in enumerate(questions, 1):
        text += f"[Q{i}] {q['type']}\n{q['title']}\n\n{q['content'].replace('<b>','').replace('</b>','').replace('<i>','').replace('</i>','').replace('<br>','\n')}\n\n"
        for opt in q['options']: text += f"{opt}\n"
        text += "\n" + "-"*40 + "\n"
    text += "\n\n🎯 정답 및 해설\n" + "="*40 + "\n\n"
    for i, q in enumerate(questions, 1): text += f"[Q{i} 정답] {q['answer']}\n[해설] {q.get('explanation', '')}\n\n"
    return text

# ---------------------------------------------------------
# 6. UI 화면 구성
# ---------------------------------------------------------
st.title("📚 영어 지문 분석 & 시험 문제 생성기 (Aikar AI 완벽판)")

col_input1, col_input2 = st.columns(2)
with col_input1:
    st.markdown("### 1. 영어 지문 입력")
    uploaded_image = st.file_uploader("📷 지문 이미지 업로드 (온라인 전용)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        with st.spinner("이미지 판독 중..."):
            txt, err = extract_text_from_image_via_ai(uploaded_image)
            if txt: st.session_state.eng_input_area = txt; st.success("추출 완료!")
            else: st.error(err)
    input_eng = st.text_area("영어 내용:", height=150, key="eng_input_area")

with col_input2:
    st.markdown("### 2. 한국어 뜻 입력")
    st.write("\n\n\n\n\n")
    input_kor = st.text_area("해석 내용 (비워두면 AI 자동 번역 / 오프라인 시 직접 입력):", height=150)

st.divider()

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    if st.button("📖 구문 학습", use_container_width=True):
        if input_eng: st.session_state.sentences, st.session_state.translations = process_inputs(input_eng, input_kor); st.session_state.mode = "learn"
with col_b2:
    if st.button("📑 단어장 & 요약", use_container_width=True):
        if input_eng:
            with st.spinner("지문 요약 및 단어장 생성 중..."):
                s, _ = process_inputs(input_eng, input_kor)
                st.session_state.summary_data = generate_summary_and_vocab(s)
                st.session_state.mode = "summary"
with col_b3:
    if st.button("🧠 암기 테스트", use_container_width=True):
        if input_eng: st.session_state.sentences, st.session_state.translations = process_inputs(input_eng, input_kor); st.session_state.mode = "memo"
with col_b4:
    if st.button("📝 5개 유형 문제 한번에 생성", use_container_width=True):
        if input_eng:
            with st.spinner("수능 5대 유형 문제 한번에 생성 중... (Lumen 추론 진행 중)"):
                s, t = process_inputs(input_eng, input_kor)
                st.session_state.sentences, st.session_state.translations = s, t
                st.session_state.generated_questions = generate_5_exam_questions(s)
                st.session_state.mode = "exam"

st.divider()

# ---------------------------------------------------------
# 7. 모드별 기능 실행
# ---------------------------------------------------------
if st.session_state.mode == "learn":
    st.subheader("📖 끊어 읽기 & 구문 학습 (TTS 음성 오디오 지원)")
    for idx, (sent, trans) in enumerate(zip(st.session_state.sentences, st.session_state.translations), 1):
        spans = check_local_rules(sent)
        fmt_eng = format_sentence_html(sent, spans)
        
        st.markdown(f"**[문장 #{idx}]**")
        st.markdown(f'<div class="sentence-container">{fmt_eng}</div>', unsafe_allow_html=True)
        render_tts_button(sent)
        with st.expander("🔍 한국어 해석 보기"): st.info(trans)

elif st.session_state.mode == "summary":
    st.subheader("📑 핵심 요약 & 필수 단어장")
    data = st.session_state.summary_data
    if data:
        st.markdown(f'<div class="summary-box"><b>📌 핵심 주제 (Topic):</b> {data.get("topic", "")}<br><br><b>💡 지문 요지 (Main Idea):</b> {data.get("main_idea", "")}</div>', unsafe_allow_html=True)
        st.markdown("### 🔠 핵심 필수 어휘")
        vocab_list = data.get("vocab", [])
        if vocab_list:
            v_html = "<table style='width:100%; text-align:left; border-collapse: collapse;'><tr><th style='border-bottom:2px solid #ccc; padding:8px;'>단어</th><th style='border-bottom:2px solid #ccc; padding:8px;'>뜻</th><th style='border-bottom:2px solid #ccc; padding:8px;'>문맥상 쓰임 / 정보</th></tr>"
            for v in vocab_list: v_html += f"<tr><td style='border-bottom:1px solid #eee; padding:8px;'><b>{v['word']}</b></td><td style='border-bottom:1px solid #eee; padding:8px;'>{v['meaning']}</td><td style='border-bottom:1px solid #eee; padding:8px; color:#555;'>{v.get('context','')}</td></tr>"
            v_html += "</table>"
            st.markdown(v_html, unsafe_allow_html=True)

elif st.session_state.mode == "memo":
    st.subheader("🧠 문장 해석 암기 테스트 (TTS 지원)")
    
    total_sents = len(st.session_state.sentences)
    answered_sents = 0
    total_score_sum = 0
    
    for idx, (sent, tgt) in enumerate(zip(st.session_state.sentences, st.session_state.translations), 1):
        ans = st.session_state.get(f"memo_{idx}", "")
        if ans:
            answered_sents += 1
            sim_score = round(SequenceMatcher(None, re.sub(r'[^\w\s]', '', ans).replace(" ",""), re.sub(r'[^\w\s]', '', tgt).replace(" ","")).ratio()*100, 1)
            total_score_sum += sim_score
            
    avg_score = round(total_score_sum / answered_sents, 1) if answered_sents > 0 else 0
    
    st.markdown(f"""
    <div class="score-box">
        📊 <b>암기 테스트 현황:</b> {answered_sents}/{total_sents} 문장 완료 | 
        🎯 <b>평균 해석 일치도: <span style="color:#e74a3b;">{avg_score}점</span></b>
    </div>
    """, unsafe_allow_html=True)
    st.write("---")

    for idx, (sent, tgt) in enumerate(zip(st.session_state.sentences, st.session_state.translations), 1):
        st.markdown(f"**Q{idx}. {sent}**")
        render_tts_button(sent, height=38)
        
        ans = st.text_input(f"Q{idx} 해석 작성:", key=f"memo_{idx}")
        if ans:
            score = round(SequenceMatcher(None, re.sub(r'[^\w\s]', '', ans).replace(" ",""), re.sub(r'[^\w\s]', '', tgt).replace(" ","")).ratio()*100, 1)
            if score >= 85: st.success(f"{score}점 (정답) 🎯")
            else: st.error(f"{score}점 (오답) 💡")
            with st.expander("정답 보기"): st.write(tgt)
        st.write("---")

elif st.session_state.mode == "exam":
    st.subheader("📝 실전 수능 유형 객관식 문제 (5대 핵심 유형)")
    
    questions = st.session_state.generated_questions
    if questions:
        total_questions = len(questions)
        answered_count = sum(1 for idx in range(1, total_questions + 1) if st.session_state.get(f"exam_{idx}") is not None)
        correct_count = sum(1 for idx, q in enumerate(questions, 1) if st.session_state.get(f"exam_{idx}") == q["answer"])
        score_percent = int((correct_count / total_questions) * 100)
        
        if answered_count == total_questions and correct_count == total_questions:
            st.balloons()
            st.success("🎉 축하합니다! 모든 문제를 맞혀 100점 만점을 기록했습니다!")

        st.markdown(f"""
        <div class="score-box">
            📊 <b>시험 현황:</b> {answered_count}/{total_questions} 문제 완료 | 
            🎯 <b>현재 점수: <span style="color:#e74a3b;">{score_percent}점</span></b> ({correct_count}/{total_questions} 정답)
        </div>
        """, unsafe_allow_html=True)
        
        dl_text = create_export_text(questions)
        st.download_button("📥 문제지 및 해설 다운로드 (.txt)", data=dl_text, file_name="English_Exam.txt", mime="text/plain")
        st.write("---")
        
        for idx, q in enumerate(questions, 1):
            st.markdown(f"### Q{idx}. {q['type']}")
            st.markdown(f"**{q['title']}**")
            st.markdown(f'<div class="question-box">{q["content"]}</div>', unsafe_allow_html=True)
            
            choice = st.radio("정답 선택:", q["options"], key=f"exam_{idx}", index=None)
            if choice:
                if choice == q["answer"]:
                    st.success("🎯 정답입니다! (+20점)")
                else:
                    st.error(f"❌ 오답입니다. (선택한 답: {choice})")
                with st.expander("🔍 정답 및 상세 해설 보기"):
                    st.write(f"**정답:** {q['answer']}\n\n**해설:** {q.get('explanation', '')}")
            st.write("---")