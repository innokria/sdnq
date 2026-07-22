
# app_openvino.py

import gc
import os
import threading

import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from huggingface_hub import (
    login,
    HfApi,
    snapshot_download,
)

import tempfile
import shutil


from sdnq import sdnq_post_load_quant

import tempfile
import shutil

from huggingface_hub import (
    HfApi,
    login,
    snapshot_download,
)

HF_TOKEN = os.getenv("HF_TOKEN")
# =====================================================
# OPENVINO DEVICES
# =====================================================

os.environ["SDNQ_USE_OPENVINO_MM"] = "1"
os.environ["SDNQ_OPENVINO_DEVICE"] = "CPU"


import importlib.metadata

print("=" * 50)

for pkg in [
    "openvino",
    "optimum",
    "optimum-intel",
    "transformers",
]:
    try:
        print(
            pkg,
            importlib.metadata.version(pkg)
        )
    except Exception as e:
        print(pkg, e)

print("=" * 50)




OPENVINO_DEVICES = ["CPU"]

try:
    import openvino

    core = openvino.Core()
    OPENVINO_DEVICES = core.available_devices

    print(
        "OpenVINO devices:",
        OPENVINO_DEVICES
    )

except Exception as e:
    print(
        "OpenVINO detection failed:",
        e
    )

import openvino

print("openvino module:", openvino.__file__)
print("openvino version:", openvino.__version__)
# =====================================================
# CONFIG
# =====================================================

MODEL_FILES = {
    "rahul-Gemma2-2B":"rahul7star/sdnq-unit4-gemma-2-2b",
    "rahul-Gemma4-E2B":"rahul7star/sdnq-8bit-gemma-4-E2B-it",

    "Qwen3-0.6B":
        "Qwen/Qwen3-0.6B",

    "Qwen3-1.7B":
        "Qwen/Qwen3-1.7B",

    "Qwen3-4B":
        "Qwen/Qwen3-4B",

    "Gemma-3-1B":
        "google/gemma-3-1b-it",

    "SmolLM3-3B":
        "HuggingFaceTB/SmolLM3-3B",

    "Qwen2.5-3B":
        "Qwen/Qwen2.5-3B-Instruct",

    "Phi-4-Mini":
        "microsoft/Phi-4-mini-instruct",

    "Llama-3.2-1B":
        "meta-llama/Llama-3.2-1B-Instruct",

    "Llama-3.2-3B":
        "meta-llama/Llama-3.2-3B-Instruct",
}


os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"



CSS = """

.gradio-container {
    max-width: 1100px !important;
}

#chatbot {
    height:650px !important;
}

footer {
    display:none !important;
}

"""



# =====================================================
# GLOBAL STATE
# =====================================================

model = None
tokenizer = None

CURRENT_BACKEND = "OPENVINO"



# =====================================================
# DEVICE
# =====================================================

def get_device():

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(
            torch.backends,
            "mps"
        )
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


DEVICE = get_device()



# =====================================================
# CLEAN MEMORY
# =====================================================





# =====================================================
# MODEL LOADER
# =====================================================

def unload_model():

    global model
    global tokenizer
    global CURRENT_BACKEND

    try:
        model = None
        tokenizer = None
        CURRENT_BACKEND = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    except Exception as e:
        print(f"Unload error: {e}")

    print("✅ Model unloaded")


# =====================================================
# HELPERS
# =====================================================

def clear_chat():
    return [], ""


def clear_input():
    return ""


# =====================================================
# MODEL LOADER
# =====================================================

def load_model(
    model_name,
    quant_type,
    backend,
):

    global model
    global tokenizer
    global CURRENT_BACKEND


    model_id = MODEL_FILES[model_name]


    # =================================================
    # Normalize backend dropdown
    # =================================================

    backend = backend.strip()

    if "SDNQ" in backend:
        backend = "SDNQ"

    elif "OpenVINO" in backend:
        backend = "OpenVINO"

    else:
        raise ValueError(
            f"Unknown backend selected: {backend}"
        )


    try:

        unload_model()


        # =================================================
        # TOKENIZER
        # =================================================

        yield (
            "Loading tokenizer...\n"
            f"{model_id}"
        )


        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )


        yield (
            "Loading model...\n"
            f"Backend: {backend}\n"
            f"{model_id}"
        )


        # =================================================
        # SDNQ BACKEND
        # =================================================

        if backend == "SDNQ":


            dtype = (
                torch.bfloat16
                if DEVICE != "cpu"
                else torch.float32
            )


            # Load model first
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )


            # ---------------------------------------------
            # Detect existing SDNQ quantization
            # ---------------------------------------------

            already_quantized = False


            try:

                # Check model modules
                for _, module in model.named_modules():

                    module_name = (
                        module.__class__
                        .__name__
                        .lower()
                    )


                    if (
                        "sdnq" in module_name
                        or "quant" in module_name
                    ):
                        already_quantized = True
                        break


                # Check config attribute
                if hasattr(
                    model,
                    "quantization_config"
                ):
                    already_quantized = True


            except Exception:
                pass



            # ---------------------------------------------
            # Existing SDNQ model
            # ---------------------------------------------

            if already_quantized:

                yield (
                    "✅ SDNQ model detected\n"
                    "Skipping quantization"
                )


            # ---------------------------------------------
            # Normal HF model
            # ---------------------------------------------

            else:

                yield (
                    "Applying SDNQ quantization...\n"
                    f"Type: {quant_type}"
                )


                model = sdnq_post_load_quant(
                    model,
                    weights_dtype=quant_type,
                    group_size=32,
                )


            CURRENT_BACKEND = "SDNQ"



        # =================================================
        # OPENVINO BACKEND
        # =================================================

        elif backend == "OpenVINO":


            yield (
                "Loading OpenVINO CPU model..."
            )


            from optimum.intel.openvino import (
                OVModelForCausalLM
            )


            model = OVModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                device="CPU",
            )


            CURRENT_BACKEND = "OpenVINO"



        # =================================================
        # FINALIZE
        # =================================================

        if hasattr(model, "eval"):
            model.eval()



        yield (
            "✅ Loaded\n\n"
            f"Model: {model_id}\n"
            f"Backend: {CURRENT_BACKEND}\n"
            f"Quant: "
            f"{quant_type if CURRENT_BACKEND == 'SDNQ' else 'N/A'}\n"
            f"Device: {DEVICE}"
        )



    except Exception as e:

        import traceback

        traceback.print_exc()


        # cleanup failed model

        model = None
        tokenizer = None
        CURRENT_BACKEND = None


        gc.collect()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        yield (
            "❌ Error\n\n"
            f"{type(e).__name__}: {e}"
        )
# =====================================================
# CHAT GENERATION
# =====================================================

def respond(
    message,
    history,
    system_prompt,
    temperature,
    max_tokens,
):
    global model
    global tokenizer

    history = history or []


    if model is None:
        history.append(
            {
                "role": "assistant",
                "content": "Please load a model first.",
            }
        )
        yield history
        return



    # =============================================
    # Build messages
    # =============================================

    messages = []


    # Some models do not support system role
    if system_prompt and system_prompt.strip():

        message = (
            
            f"{message}"
        )


    messages.extend(history)


    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )



    # =============================================
    # Chat template
    # =============================================

    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


    except Exception as e:

        print(
            "Chat template failed:",
            e
        )

        # fallback plain prompt

        prompt = ""

        for m in messages:
            prompt += (
                f"{m['role'].upper()}: "
                f"{m['content']}\n"
            )

        prompt += "ASSISTANT:"



    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )


    if DEVICE != "cpu":

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }



    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )


    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        do_sample=True,
        temperature=float(temperature),
        max_new_tokens=int(max_tokens),
        pad_token_id=(
            tokenizer.eos_token_id
        ),
    )


    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True,
    )

    thread.start()



    history.append(
        {
            "role": "user",
            "content": message,
        }
    )


    history.append(
        {
            "role": "assistant",
            "content": "",
        }
    )



    partial = ""


    for token in streamer:

        partial += token

        history[-1]["content"] = partial

        yield history



# =====================================================
# Quant and Upload
# =====================================================

HF_TOKEN = os.getenv("HF_TOKEN")





def quantize_and_upload(
    source_model,
    output_repo,
    preset,
    quant_dtype,
    group_size,
    use_hadamard,
    use_svd,
    use_quantized_matmul,
    load_dtype,
    quantized_matmul_dtype,
):

    try:

        if not HF_TOKEN:
            yield "❌ HF_TOKEN secret not found"
            return

        yield "🔑 Logging into Hugging Face..."

        login(token=HF_TOKEN)

        if not output_repo.strip():

            model_name = source_model.split("/")[-1]

            output_repo = (
                f"rahul7star/"
                f"{model_name}-sdnq-{quant_dtype}"
            )

        if preset == "Fast":

            quant_dtype = "uint4"
            group_size = 32

        elif preset == "Balanced":

            quant_dtype = "int8"
            group_size = 32

        elif preset == "Quality":

            quant_dtype = "float16"
            group_size = -1

        yield f"📥 Downloading\n{source_model}"

        workdir = tempfile.mkdtemp()

        model_dir = os.path.join(
            workdir,
            "model",
        )

        output_dir = os.path.join(
            workdir,
            "quantized",
        )

        snapshot_download(
            repo_id=source_model,
            local_dir=model_dir,
        )

        yield "📚 Loading tokenizer..."

        tokenizer_q = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        selected_dtype = dtype_map.get(
            load_dtype,
            "auto",
        )

        yield (
            f"🧠 Loading model\n"
            f"DType: {load_dtype}"
        )

        model_q = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=selected_dtype,
            low_cpu_mem_usage=True,
        )

        yield (
            f"⚡ Quantizing\n"
            f"Type: {quant_dtype}"
        )

        kwargs = dict(
            weights_dtype=quant_dtype,
            group_size=int(group_size),
            use_hadamard=bool(use_hadamard),
            use_svd=bool(use_svd),
            dequantize_fp32=True,
        )

        if use_quantized_matmul:

            kwargs["use_quantized_matmul"] = True

            if (
                quantized_matmul_dtype
                != "disabled"
            ):
                kwargs[
                    "quantized_matmul_dtype"
                ] = quantized_matmul_dtype

        model_q = sdnq_post_load_quant(
            model_q,
            **kwargs,
        )

        yield "💾 Saving quantized model..."

        model_q.save_pretrained(
            output_dir
        )

        tokenizer_q.save_pretrained(
            output_dir
        )

        del model_q
        gc.collect()

        yield "🚀 Creating HF repo..."

        api = HfApi()

        api.create_repo(
            repo_id=output_repo,
            repo_type="model",
            exist_ok=True,
        )

        yield "⬆️ Uploading files..."

        api.upload_folder(
            folder_path=output_dir,
            repo_id=output_repo,
            repo_type="model",
        )

        shutil.rmtree(
            workdir,
            ignore_errors=True,
        )

        yield (
            "✅ Complete\n\n"
            f"Repo:\n"
            f"https://huggingface.co/{output_repo}"
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        yield (
            "❌ Error\n\n"
            f"{type(e).__name__}: {e}"
        )
# =====================================================
# GRADIO UI
# =====================================================

# =====================================================
# GRADIO UI
# =====================================================

with gr.Blocks(
    title="SDNQ + OpenVINO CPU Chat",
    css=CSS,
) as demo:

    gr.Markdown(
        "# 🚀 SDNQ + OpenVINO CPU Chat"
    )

    with gr.Tabs():

        # =================================================
        # CHAT TAB
        # =================================================

        with gr.Tab("💬 Chat"):

            with gr.Row():

                with gr.Column(scale=5):

                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        height=650,
                    )

                    with gr.Row():

                        msg_input = gr.Textbox(
                            placeholder="Ask something...",
                            scale=5,
                            show_label=False,
                        )

                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                        )

                    gr.Examples(
                        examples=[
                            ["Explain quantum computing"],
                            ["Write a Python function"],
                            ["What is machine learning?"],
                            ["Tell me a joke"],
                            ["How does SDNQ work?"],
                            ["Write a React component"],
                        ],
                        inputs=[msg_input],
                        label="💡 Examples",
                    )

                with gr.Column(scale=2):

                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_FILES.keys()),
                        value="Qwen3-1.7B",
                        label="Model",
                    )

                    backend_dropdown = gr.Dropdown(
                        choices=[
                            "SDNQ + OpenVINO CPU",
                            "SDNQ CPU",
                            "SDNQ GPU",
                        ],
                        value="SDNQ + OpenVINO CPU",
                        label="Backend",
                    )

                    quant_dropdown = gr.Dropdown(
                        choices=[
                            "uint4",
                            "int4",
                            "uint8",
                            "int8",
                        ],
                        value="uint4",
                        label="SDNQ Quant",
                    )

                    load_btn = gr.Button(
                        "⚡ Load Model",
                        variant="primary",
                    )

                    status = gr.Textbox(
                        label="Status",
                        lines=6,
                    )

                    system_prompt = gr.Textbox(
                        value="You are a helpful assistant.",
                        label="System Prompt",
                    )

                    temperature = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )

                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=4096,
                        value=512,
                        step=32,
                        label="Max Tokens",
                    )

                    clear_btn = gr.Button(
                        "🧹 Clear Chat"
                    )

        # =================================================
        # QUANT TAB
        # =================================================

        with gr.Tab("📦 Quant Model"):

            gr.Markdown(
                """
# 📦 SDNQ Quantizer

Quantize Your Model


"""
            )

            quant_source_model = gr.Textbox(
                label="Source Model",
                placeholder="google/gemma-2-2b-it",
                value="google/gemma-2-2b-it",
            )

            quant_output_repo = gr.Textbox(
                label="Output Repo (optional)",
                placeholder="rahul7star/gemma-2-2b-it-sdnq",
                value="",
            )

            quant_preset = gr.Dropdown(
                choices=[
                    "Custom",
                    "Fast",
                    "Balanced",
                    "Quality",
                ],
                value="Balanced",
                label="Preset",
            )

            with gr.Row():

                quant_dtype_ui = gr.Dropdown(
                    choices=[
                        "uint2",
                        "int2",
                        "uint3",
                        "int3",
                        "uint4",
                        "int4",
                        "uint5",
                        "int5",
                        "uint6",
                        "int6",
                        "uint8",
                        "int8",
                        "float8_e4m3fn",
                        "float16",
                    ],
                    value="uint4",
                    label="Weights DType",
                )

                quant_group_size = gr.Slider(
                    minimum=-1,
                    maximum=256,
                    value=32,
                    step=1,
                    label="Group Size",
                )

            with gr.Row():

                quant_hadamard = gr.Checkbox(
                    value=False,
                    label="Hadamard Rotation",
                )

                quant_svd = gr.Checkbox(
                    value=False,
                    label="SVD Quantization",
                )

                quant_qmm = gr.Checkbox(
                    value=True,
                    label="Quantized MatMul",
                )

            with gr.Accordion(
                "Advanced Settings",
                open=False,
            ):

                load_dtype = gr.Dropdown(
                    choices=[
                        "auto",
                        "float32",
                        "float16",
                        "bfloat16",
                    ],
                    value="auto",
                    label="Load DType",
                )

                quantized_matmul_dtype = gr.Dropdown(
                    choices=[
                        "int8",
                        "float16",
                        "disabled",
                    ],
                    value="int8",
                    label="MatMul DType",
                )

            quant_btn = gr.Button(
                "🚀 Quantize & Upload",
                variant="primary",
            )

            quant_logs = gr.Textbox(
                label="Logs",
                lines=25,
            )

    # =================================================
    # EVENTS
    # =================================================

    load_btn.click(
        load_model,
        inputs=[
            model_dropdown,
            quant_dropdown,
            backend_dropdown,
        ],
        outputs=status,
    )

    quant_btn.click(
        quantize_and_upload,
        inputs=[
            quant_source_model,
            quant_output_repo,
            quant_preset,
            quant_dtype_ui,
            quant_group_size,
            quant_hadamard,
            quant_svd,
            quant_qmm,
            load_dtype,
            quantized_matmul_dtype,
        ],
        outputs=quant_logs,
    )

    send_event = send_btn.click(
        respond,
        inputs=[
            msg_input,
            chatbot,
            system_prompt,
            temperature,
            max_tokens,
        ],
        outputs=chatbot,
    )

    submit_event = msg_input.submit(
        respond,
        inputs=[
            msg_input,
            chatbot,
            system_prompt,
            temperature,
            max_tokens,
        ],
        outputs=chatbot,
    )

    send_event.then(
        clear_input,
        outputs=msg_input,
    )

    submit_event.then(
        clear_input,
        outputs=msg_input,
    )

    clear_btn.click(
        clear_chat,
        outputs=[
            chatbot,
            msg_input,
        ],
    )


# =====================================================
# START
# =====================================================



# =====================================================
# START
# =====================================================

demo.queue(
    max_size=10
)


demo.launch(
    server_name="0.0.0.0",
)
 
