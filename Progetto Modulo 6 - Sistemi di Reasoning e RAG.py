import os
import json
import re
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, util

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================
# FASE 1 - MODELLO LLAMA 3 QUANTIZZATO A 4 BIT
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("Per questa configurazione è necessaria una GPU CUDA.")

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=compute_dtype
)

model.eval()

SYSTEM_PROMPT = """Sei un assistente AI tecnico, preciso e affidabile.
Segui attentamente le istruzioni dell'utente.
Non inventare informazioni e distingui chiaramente fatti e ipotesi.
Quando affronti un problema complesso, ragiona passo dopo passo internamente e restituisci una risposta finale accompagnata solo da una breve spiegazione.
Quando ricevi del contesto da un sistema RAG, utilizzalo come fonte principale."""

def prepare_inputs(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    return inputs.to(next(model.parameters()).device)

def generate(messages, temperature=0.2, top_p=0.9, max_new_tokens=256):
    inputs = prepare_inputs(messages)

    generation_args = {
        **inputs,
        "max_new_tokens":max_new_tokens,
        "pad_token_id":tokenizer.eos_token_id,
        "do_sample":temperature > 0
    }

    if temperature > 0:
        generation_args.update({"temperature":temperature,"top_p":top_p})

    with torch.inference_mode():
        outputs = model.generate(**generation_args)

    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# FASE 2 - PROMPT ENGINEERING E REASONING
# ============================================================

def reason(question):
    prompt = f"""<ISTRUZIONI>
Analizza attentamente il problema.
Pensa al problema passo dopo passo prima di formulare la risposta.
Non mostrare una lunga catena di ragionamento: restituisci soltanto:
1. la risposta finale;
2. una breve spiegazione dei passaggi essenziali.
</ISTRUZIONI>
<INPUT_UTENTE>
{question}
</INPUT_UTENTE>"""

    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":prompt}
    ]

    return generate(messages, temperature=0.1, max_new_tokens=300)


# ============================================================
# FASE 3 - RETRIEVAL-AUGMENTED GENERATION
# ============================================================

documents = [
    {
        "source": "quantization.md",
        "text":"La quantizzazione riduce la precisione numerica dei pesi di un modello per diminuire il consumo di memoria. La quantizzazione a 4 bit permette di utilizzare modelli molto grandi occupando molta meno VRAM."
    },
    {
        "source": "rag.md",
        "text":"Un sistema RAG è composto principalmente dalle fasi Retrieve, Augment e Generate. Prima recupera i documenti semanticamente più pertinenti, poi inserisce le informazioni recuperate nel prompt e infine genera una risposta utilizzando quel contesto."
    },
    {
        "source":"embeddings.md",
         "text":"Gli embedding rappresentano testi come vettori numerici. Testi semanticamente simili tendono ad avere vettori vicini nello spazio degli embedding. La cosine similarity può essere utilizzata per misurare questa somiglianza."
    },
    {
         "source":"lora.md",
        "text":"LoRA è una tecnica di parameter-efficient fine-tuning. Mantiene congelati i pesi principali del modello e addestra piccole matrici low-rank aggiuntive, riducendo memoria e costo computazionale."
    }
]



def chunk_text(text, chunk_size=60, overlap=10):
    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
        if words[i:i + chunk_size]
    ]


def build_chunks(documents):
    chunks = []

    for document in documents:
        for i, text in enumerate(chunk_text(document["text"])):
            chunks.append({
                "source":document["source"],
                "chunk_id":i,
                "text":text
            })
    return chunks

chunks = build_chunks(documents)

# CPU per gli embedding: lasciamo più VRAM possibile a Llama
embedder = SentenceTransformer(EMBED_MODEL_ID, device="cpu")

chunk_embeddings = embedder.encode(
    [chunk["text"] for chunk in chunks],
    convert_to_tensor=True,
    normalize_embeddings=True
)

def retrieve(query, k=3):
    query_embedding = embedder.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    k = min(k, len(chunks))

    values, indices = torch.topk(scores, k=k)

    return [ {**chunks[index],"score":float(score)}
            for score, index in zip(values.tolist(), indices.tolist())]

def rag_answer(question, k=3):
    retrieved = retrieve(question, k)

    context = "\n\n".join(
        f"[Fonte: {item['source']} | similarity={item['score']:.3f}]\n{item['text']}"
        for item in retrieved )
    prompt = f"""<CONTESTO>

{context}
</CONTESTO>

<DOMANDA>
{question}
</DOMANDA>

<ISTRUZIONI>
Rispondi utilizzando le informazioni presenti nel CONTESTO.
Se il contesto non contiene informazioni sufficienti, dichiaralo esplicitamente.
Non inventare informazioni mancanti.
</ISTRUZIONI>"""

    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":prompt}
    ]

    answer = generate(messages, temperature=0.1, max_new_tokens=300)
    return {
        "answer":answer,
        "retrieved_chunks":retrieved
    }


# ============================================================
# FASE 4 - ZERO-SHOT SENTIMENT ANALYSIS
# ============================================================

def sentiment_prompt(text):
    return f"""<TASK>
Classifica il sentimento del testo senza esempi e senza addestramento aggiuntivo.
Le uniche classi ammesse sono:
- Positivo
- Negativo

Restituisci esclusivamente JSON valido:
{{"sentiment":"Positivo"}}
</TASK>
<TEXT>
{text}
</TEXT>"""

def extract_json(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Nessun JSON trovato nell'output: {text}")

    return json.loads(match.group())

def sentiment_batch(texts):
    conversations = [
        [{"role":"system","content":SYSTEM_PROMPT},
          {"role":"user","content":sentiment_prompt(text)}]
        for text in texts
    ]

    formatted_prompts = [
        tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False)
        for conversation in conversations
    ]

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(next(model.parameters()).device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    results = []

    for text, output in zip(texts, decoded):
        data = extract_json(output)

        if data.get("sentiment") not in {"Positivo","Negativo"}:
            raise ValueError(f"Sentiment non valido: {data}")

        results.append({
            "text":text,
            "sentiment":data["sentiment"]})

    return results

def sentiment_zero_shot(text):
    return sentiment_batch([text])[0]


# ============================================================
# STREAMING
# ============================================================

def stream_generate(messages, temperature=0.2, top_p=0.9, max_new_tokens=256):
    inputs = prepare_inputs(messages)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True)

    generation_args = {
        **inputs,
        "streamer":streamer,
        "max_new_tokens":max_new_tokens,
        "pad_token_id":tokenizer.eos_token_id,
        "do_sample":temperature > 0}

    if temperature > 0:
        generation_args.update({"temperature":temperature,"top_p":top_p})

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    for piece in streamer:
        yield piece

    thread.join()

# ============================================================
# FASE 5 - OUTPUT STRUTTURATO JSON
# ============================================================


def structured_answer(question):
    prompt = f"""<TASK>
Rispondi alla domanda e restituisci ESCLUSIVAMENTE un singolo oggetto JSON valido.

Non utilizzare Markdown, backtick o testo al di fuori del JSON.

Schema obbligatorio:
{{
    "answer":"stringa",
    "confidence":0.0,
    "keywords":["stringa"]
}}

Vincoli:
- "answer" deve essere una stringa;
- "confidence" deve essere un numero tra 0 e 1;
- "keywords" deve contenere da 1 a 5 elementi.
</TASK>

<QUESTION>
{question}
</QUESTION>"""

    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":prompt}
    ]

    raw_output = generate(
        messages,
        temperature=0,
        max_new_tokens=180
    )

    data = extract_json(raw_output)

    required_keys = {"answer","confidence","keywords"}

    if set(data.keys()) != required_keys:
        raise ValueError(f"Schema JSON non valido: {data}")

    if not isinstance(data["answer"], str):
        raise TypeError("'answer' deve essere una stringa.")

    if not isinstance(data["confidence"], (int, float)):
        raise TypeError("'confidence' deve essere numerico.")

    if not isinstance(data["keywords"], list):
        raise TypeError("'keywords' deve essere una lista.")

    if not 0 <= data["confidence"] <= 1:
        raise ValueError("'confidence' deve essere compreso tra 0 e 1.")

    if not 1 <= len(data["keywords"]) <= 5:
        raise ValueError("'keywords' deve contenere da 1 a 5 elementi.")

    return data


# ============================================================
# TEST DELLA PIPELINE
# ============================================================

print("\n--- REASONING ---")
question = "Un'auto percorre 150 km in 2 ore e 30 minuti. Qual è la velocità media?"
print(reason(question))
print("\n--- RAG ---")

rag_result = rag_answer(
    "A cosa servono gli embedding in un sistema RAG?")

print(rag_result["answer"])
print("\nChunk recuperati:")

for chunk in rag_result["retrieved_chunks"]:
    print(f"- {chunk['source']} | similarity={chunk['score']:.3f}")
print("\n--- ZERO-SHOT SENTIMENT ---")

reviews = [
    "Il prodotto è fantastico e funziona perfettamente.",
    "Esperienza pessima, si è rotto dopo un giorno.",
    "Sono molto soddisfatto dell'acquisto.",
    "Non comprerei mai più questo prodotto."
]

results = sentiment_batch(reviews)
print(json.dumps(results, ensure_ascii=False, indent=2))

print("\n--- STREAMING ---")

messages = [
    {"role":"system","content":SYSTEM_PROMPT},
    {"role":"user","content":"Spiegami in poche righe cos'è un sistema RAG."}
]

for token in stream_generate(messages, temperature=0.2, max_new_tokens=120):
    print(token, end="", flush=True)

print("\n\n--- JSON STRUTTURATO ---")
result = structured_answer(
    "Spiegami brevemente cos'è LoRA."
)
print(json.dumps(result, ensure_ascii=False, indent=2))