# 🚀 Complete Guide: Publishing & Deploying Invoice Qwen-VL LoRA on Hugging Face

This guide provides end-to-end instructions for uploading your fine-tuned LoRA model weights to your Hugging Face account (**`yashupadhyay`**) and connecting it with your **Hydro HRMS & AI Invoice Extractor platform**.

---

## 📌 Repository & Model Overview

| Attribute | Details |
|---|---|
| **Base Model** | `Qwen/Qwen2.5-VL-7B-Instruct` (Unsloth 4-bit / 16-bit) |
| **Adapter Type** | PEFT LoRA (Rank: 16, Alpha: 16, Safetensors: 190.4 MB) |
| **Task** | Multimodal Vision-Language Invoice & Receipt Extraction to JSON |
| **Hugging Face Account** | `yashupadhyay` |
| **GitHub Repository** | [YashUpadhyay5/invoice-qwen-vl-lora](https://github.com/YashUpadhyay5/invoice-qwen-vl-lora) |
| **Local Folder** | `C:\Users\DELL\Downloads\invoice-qwen-vl-lora` |

---

## 🔑 Step 1: Generate a Hugging Face Access Token (30 Seconds)

To publish models to Hugging Face, an access token with **Write** permission is required:

1. Open your browser and go to: **[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**
2. Click **Create new token**.
3. Fill in the details:
   - **Token Name**: `invoice-upload`
   - **Token Type**: Select **Write** (or check *Repositories $\rightarrow$ Create, edit, and post content*).
4. Click **Create token** and copy the generated token (starts with `hf_...`).

---

## ⚡ Step 2: Upload the Model to Hugging Face

All model weights and configuration files are ready in `C:\Users\DELL\Downloads\invoice-qwen-vl-lora`. Choose either method:

### Option A: Automated Script (Fastest & Easiest)

1. Open a PowerShell or Command Prompt terminal.
2. Run the prepared upload script:
   ```powershell
   cd "C:\Users\DELL\Downloads\invoice-qwen-vl-lora"
   python upload_to_hf.py
   ```
   *(Or simply double-click **`upload_to_hf.bat`**)*.
3. Paste your Hugging Face token when prompted.
4. The script will automatically:
   - Authenticate your account (`yashupadhyay`)
   - Create the repository `yashupadhyay/invoice-qwen-vl-lora` on Hugging Face
   - Upload all model weights (`adapter_model.safetensors`), configs, and tokenizer files.

---

### Option B: Upload via Git CLI

If you prefer standard Git:

1. Go to [https://huggingface.co/new](https://huggingface.co/new)
2. **Owner**: `yashupadhyay`
3. **Model name**: `invoice-qwen-vl-lora`
4. Select **Public** or **Private**, then click **Create Model**.
5. In your terminal, run:
   ```powershell
   cd "C:\Users\DELL\Downloads\invoice-qwen-vl-lora"
   git remote add hf https://huggingface.co/yashupadhyay/invoice-qwen-vl-lora
   git push hf main
   ```
   *(When prompted, enter username `yashupadhyay` and your `hf_...` token as the password).*

---

## 🌐 Step 3: Deploy as a Free Interactive Web Demo (Hugging Face Spaces + ZeroGPU)

You can run live inference for free using Hugging Face's ZeroGPU infrastructure:

1. In Hugging Face, click **New Space** ([https://huggingface.co/new-space](https://huggingface.co/new-space)).
2. Name: `invoice-extractor-demo`
3. Space SDK: **Gradio**
4. Space Hardware: **ZeroGPU** (Free A100 GPU compute provided by Hugging Face).
5. Add the following 2 files in the Space:

### `requirements.txt`:
```txt
torch
transformers>=4.49.0
peft>=0.14.0
accelerate
qwen-vl-utils
pillow
gradio
spaces
```

### `app.py`:
```python
import gradio as gr
import torch
from PIL import Image
import spaces
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_MODEL_ID = "yashupadhyay/invoice-qwen-vl-lora"

print("Loading Qwen 2.5-VL base model...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Attaching fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
model.eval()

@spaces.GPU(duration=60)
def extract_invoice(image: Image.Image, instruction: str = "Extract invoice information into JSON."):
    if image is None:
        return "Please upload an invoice image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1536)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

demo = gr.Interface(
    fn=extract_invoice,
    inputs=[
        gr.Image(type="pil", label="Upload Invoice"),
        gr.Textbox(value="Extract invoice information into JSON.", label="Prompt Instruction")
    ],
    outputs=gr.Code(language="json", label="Structured Extracted Output"),
    title="Invoice Extractor (Qwen 2.5-VL 7B + LoRA)",
    description="Fine-tuned multimodal AI model for extracting vendor, invoice, line items, and tax summaries."
)

if __name__ == "__main__":
    demo.launch()
```

---

## 🔗 Step 4: Connecting the Hugging Face Model to Hydro HRMS

Once your Hugging Face Space or Dedicated Endpoint is running, connect it to your local HRMS backend:

In `backend/.env`:
```env
# Point OCR service directly to your Hugging Face API
OCR_API_URL=https://yashupadhyay-invoice-extractor-demo.hf.space/api/predict
RUNPOD_API_KEY=hf_your_hugging_face_token
```

---

## 📁 File Structure in your GitHub & Local Model Repo

```
invoice-qwen-vl-lora/
├── .gitattributes          # Configured with Git LFS for large model files
├── adapter_config.json     # LoRA adapter hyperparameters (Rank 16, Alpha 16)
├── adapter_model.safetensors # 190.4 MB fine-tuned weights
├── chat_template.jinja     # Qwen chat formatting template
├── processor_config.json   # Multimodal image processor configuration
├── tokenizer.json          # Tokenizer vocabulary
├── tokenizer_config.json   # Tokenizer settings
├── README.md               # Model documentation card
├── upload_to_hf.py         # Automated Hugging Face upload script
├── upload_to_hf.bat        # Windows 1-click launcher
└── HUGGING.md              # This complete deployment guide
```
