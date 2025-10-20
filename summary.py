from typing import Optional
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def summarize_content(raw_text: str, model_name: str = "qwen-flash") -> str:
    """
    Summarize raw extracted web content into a concise, factual digest using a single-turn OpenAI-compatible Qwen call.

    - Keep key facts, numbers, names, and dates that are likely useful for reasoning.
    - Avoid long quotes; do not include boilerplate, navigation, or ads.
    """
    if not raw_text or not raw_text.strip():
        return "(empty content)"

    # Use a system prompt instead of concatenating instruction and raw text
    system_prompt = (
        "You are a helpful assistant that summarizes web page content for a browsing agent.\n"
        "Produce a detailed summary.\n"
        "Requirements:\n"
        "- Keep critical facts, numbers, dates, entities, and claims.\n"
        "- No long quotes; no boilerplate, navigation, or unrelated content.\n"
        "- Output should be standalone and easy to scan.\n"
    )

    # Treat raw_text as the user prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_text.strip()},
    ]

    # Call Qwen via the OpenAI-compatible SDK (DashScope compatible endpoint)
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    summary_text = (resp.choices[0].message.content or "").strip()
    header="url raw content too long, here is the summary:"
    return f"{header}\n{summary_text}"


# write code to test the function
if __name__ == "__main__":
    raw_text = "The field of 4D world modeling - aiming to jointly capture spatial geometry and temporal dynamics - has witnessed remarkable progress in recent years, driven by advances in large-scale generative models and multimodal learning. However, the development of truly general 4D world models remains fundamentally constrained by the availability of high-quality data. Existing datasets and benchmarks often lack the dynamic complexity, multi-domain diversity, and spatial-temporal annotations required to support key tasks such as 4D geometric reconstruction, future prediction, and camera-control video generation. To address this gap, we introduce OmniWorld, a large-scale, multi-domain, multi-modal dataset specifically designed for 4D world modeling. OmniWorld consists of a newly collected OmniWorld-Game dataset and several curated public datasets spanning diverse domains. Compared with existing synthetic datasets, OmniWorld-Game provides richer modality coverage, larger scale, and more realistic dynamic interactions. Based on this dataset, we establish a challenging benchmark that exposes the limitations of current state-of-the-art (SOTA) approaches in modeling complex 4D environments. Moreover, fine-tuning existing SOTA methods on OmniWorld leads to significant performance gains across 4D reconstruction and video generation tasks, strongly validating OmniWorld as a powerful resource for training and evaluation. We envision OmniWorld as a catalyst for accelerating the development of general-purpose 4D world models, ultimately advancing machines' holistic understanding of the physical world."
    model_name = "qwen-flash"
    print(summarize_content(raw_text, model_name))