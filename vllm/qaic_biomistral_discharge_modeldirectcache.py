# ---------------------------------------------------------------------------------------
# Qualcomm QAIC vLLM Example: Generate Discharge Summaries (BioMistral-7B)
# ---------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
from datetime import date
from textwrap import dedent

# ----------------------------
# 1) Your patient data (example)
#    Replace this with your real data, or load from a JSON file.
# ----------------------------
patients = [
    {
        "patient_id": "ABC123",
        "name": "J. Doe",
        "age": 58,
        "sex": "Male",
        "admit_date": "2025-08-30",
        "discharge_date": str(date.today()),
        "chief_complaint": "Chest pain and shortness of breath.",
        "history_of_present_illness": (
            "Onset of substernal chest pain 6 hours prior to presentation, radiating to left arm, "
            "associated with diaphoresis and dyspnea. No prior similar episodes reported."
        ),
        "past_medical_history": [
            "Hypertension", "Hyperlipidemia", "Type 2 diabetes mellitus"
        ],
        "allergies": ["No known drug allergies"],
        "medications_home": [
            "Lisinopril 10 mg daily",
            "Atorvastatin 40 mg nightly",
            "Metformin 1000 mg BID"
        ],
        "vitals_on_admission": {"BP": "162/94", "HR": 102, "RR": 20, "Temp": "37.0 C", "SpO2": "95%"},
        "key_labs": {
            "Troponin I": "0.34 ng/mL (elevated)",
            "HbA1c": "8.2%",
            "LDL": "142 mg/dL"
        },
        "imaging": [
            "ECG: ST depression in V4-V6.",
            "CXR: No acute cardiopulmonary process."
        ],
        "hospital_course": (
            "Admitted to telemetry. Managed as NSTEMI. Started on "
            "aspirin, heparin infusion, beta-blocker, high-intensity statin. "
            "Cardiology consulted; underwent coronary angiography showing 70% LAD stenosis."
        ),
        "procedures": ["Diagnostic coronary angiography"],
        "treatments": [
            "Aspirin 325 mg load then 81 mg daily",
            "Heparin infusion per protocol",
            "Metoprolol 25 mg BID",
            "Atorvastatin 80 mg nightly",
        ],
        "discharge_medications": [
            "Aspirin 81 mg daily",
            "Ticagrelor 90 mg BID",
            "Metoprolol 25 mg BID",
            "Atorvastatin 80 mg nightly",
            "Continue Metformin 1000 mg BID",
        ],
        "condition_at_discharge": "Hemodynamically stable, chest-pain free, ambulating without assistance.",
        "follow_up_needs": [
            "Cardiology clinic in 1–2 weeks",
            "Primary care in 1 week for BP and diabetes management",
            "Cardiac rehab referral"
        ],
        "discharge_instructions": [
            "Chest pain precautions; go to ER if pain recurs or worsens",
            "Low-sodium, heart-healthy diet",
            "Medication adherence emphasized"
        ]
    }
]

# ----------------------------
# 2) Prompt builder
# ----------------------------
SYSTEM_PROMPT = dedent("""\
    You are a clinical documentation assistant. Generate a concise, well-structured discharge summary
    using ONLY the provided patient information. Do not invent facts. Use the exact section headers:

    1) DIAGNOSIS
    2) TREATMENT
    3) FOLLOW-UP

    Write in clear, professional language suitable for the medical record.
    If a field is missing, omit it gracefully without speculation.
""")

def build_prompt(p):
    # Convert structured patient data to a readable context block
    context = dedent(f"""\
        Patient ID: {p.get('patient_id','')}
        Name: {p.get('name','')}
        Age/Sex: {p.get('age','')} / {p.get('sex','')}
        Admission Date: {p.get('admit_date','')}
        Discharge Date: {p.get('discharge_date','')}
        Chief Complaint: {p.get('chief_complaint','')}
        History of Present Illness: {p.get('history_of_present_illness','')}
        Past Medical History: {", ".join(p.get('past_medical_history', []))}
        Allergies: {", ".join(p.get('allergies', []))}
        Home Medications: {", ".join(p.get('medications_home', []))}
        Vitals on Admission: {p.get('vitals_on_admission', {})}
        Key Labs: {p.get('key_labs', {})}
        Imaging: {", ".join(p.get('imaging', []))}
        Hospital Course: {p.get('hospital_course','')}
        Procedures: {", ".join(p.get('procedures', []))}
        Treatments: {", ".join(p.get('treatments', []))}
        Discharge Medications: {", ".join(p.get('discharge_medications', []))}
        Condition at Discharge: {p.get('condition_at_discharge','')}
        Follow-up Needs: {", ".join(p.get('follow_up_needs', []))}
        Discharge Instructions: {", ".join(p.get('discharge_instructions', []))}
    """)

    # Final instruction to force the three sections
    task = dedent("""\
        Using the patient context above, produce ONLY these three sections:

        1) DIAGNOSIS
        - Primary diagnosis and relevant secondary diagnoses.
        2) TREATMENT
        - Key hospital treatments/procedures and discharge medications.
        3) FOLLOW-UP
        - Specific appointments, monitoring, lifestyle instructions, and red flags.

        Keep it concise and factual.
    """)

    # A simple chat-style template
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Patient Context:\n{context}\n\n{task}\n[/INST]"
    )
    return prompt

# ----------------------------
# 3) QAIC + BioMistral-7B setup
# ----------------------------
llm = LLM(
    model="BioMistral/BioMistral-7B",
    device="qaic",
    device_group=[0],
    max_model_len=4096,
    max_seq_len_to_capture=1024,
    max_num_seqs=2,            # tune based on card capacity
    quantization="mxfp6",      # recommended for QAIC
    kv_cache_dtype="mxint8",   # saves memory
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
    disable_log_stats=False,
)

# Deterministic decoding for clinical text
sampling = SamplingParams(
    temperature=0.0,
    max_tokens=300,
    stop=None  # you can add custom stop strings if needed
)

# ----------------------------
# 4) Generate discharge summaries
# ----------------------------
prompts = [build_prompt(p) for p in patients]
outputs = llm.generate(prompts, sampling)

# ----------------------------
# 5) Print results
# ----------------------------
for out in outputs:
    pid = "UNKNOWN"
    try:
        # Quick way to show which patient this belongs to
        for line in out.prompt.splitlines():
            if line.startswith("Patient ID:"):
                pid = line.split("Patient ID:", 1)[1].strip()
                break
    except Exception:
        pass

    print("\n" + "=" * 90)
    print(f"Discharge Summary for Patient ID: {pid}")
    print("-" * 90)
    print(out.outputs[0].text.strip())
    print("=" * 90)
