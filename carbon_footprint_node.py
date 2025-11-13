# Copyright 2023 SustainML Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SustainML Carbon Footprint Node Implementation."""

from sustainml_py.nodes.CarbonFootprintNode import CarbonFootprintNode

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

# Manage signaling
import json
import multiprocessing
import numpy as np
import onnxruntime as ort
import os
import signal
import threading
import time
import torch
import transformers

# Whether to go on spinning or interrupt
running = False


# --- CarbonTracker log parser helper + debug ---
def _parse_tracker_logs_debug(log_dir):
    """
    Returns (carbon_g, energy_kwh, ci_g_per_kwh), any may be None if not found.
    Emits debug about what it read.
    """
    carbon_g = None
    energy_kwh = None
    ci_g_per_kwh = None

    try:
        print(f"[CT DEBUG] listing log_dir={log_dir}")
        for p in os.listdir(log_dir):
            print("  -", p)
    except Exception as e:
        print("[CT DEBUG] listdir failed:", e)

    try:
        logs = parser.parse_all_logs(log_dir=log_dir)
    except Exception as e:
        print("[CT DEBUG] parse_all_logs error:", e)
        logs = None

    if not logs:
        print("[CT DEBUG] no logs parsed")
        return carbon_g, energy_kwh, ci_g_per_kwh

    # Walk from end to find the richest *non-zero* entry
    for i, entry in enumerate(reversed(logs)):
        pred = entry.get("pred") or {}
        actual = entry.get("actual") or {}

        local_carbon = None
        local_energy = None

        # Try ACTUAL first, prefer non-zero
        for k in ("co2eq (g)", "co2_eq (g)", "co2 (g)"):
            if k in actual:
                v = float(actual[k])
                if v > 0.0:
                    local_carbon = v
                    break

        for k in ("energy (kWh)", "energy_kwh"):
            if k in actual:
                v = float(actual[k])
                if v > 0.0:
                    local_energy = v
                    break

        # If nothing useful in actual, try PRED
        if local_carbon is None:
            for k in ("co2eq (g)", "co2_eq (g)", "co2 (g)"):
                if k in pred:
                    v = float(pred[k])
                    if v > 0.0:
                        local_carbon = v
                        break

        if local_energy is None:
            for k in ("energy (kWh)", "energy_kwh"):
                if k in pred:
                    v = float(pred[k])
                    if v > 0.0:
                        local_energy = v
                        break

        # Carbon intensity (gCO2/kWh) if available (field name varies) – we allow 0 here
        if ci_g_per_kwh is None:
            for k in ("avg_ci (gCO2/kWh)", "carbon_intensity (gCO2/kWh)", "ci (gCO2/kWh)"):
                if k in entry:
                    ci_g_per_kwh = float(entry[k])
                    break

        # If this entry has any non-zero carbon or energy, use it and stop
        if (local_carbon is not None and local_carbon > 0.0) or (local_energy is not None and local_energy > 0.0):
            carbon_g = local_carbon
            energy_kwh = local_energy
            break

    print(f"[CT DEBUG] parsed -> carbon_g={carbon_g} energy_kwh={energy_kwh} ci_g_per_kwh={ci_g_per_kwh}")
    return carbon_g, energy_kwh, ci_g_per_kwh


# Load generic ml model and generate its input
def load_any_model(model_name, hf_token=None, unsupported_models=None, **kwargs):

    model = None

    try:
        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model configuration loaded: {config}")
        model_class = transformers.AutoModel._model_mapping.get(type(config), None)

        if unsupported_models is not None:
            for unsupported in unsupported_models:
                if unsupported.lower() in model_class.__name__.lower():
                    raise ValueError(f"[WARNING] Models that use '{unsupported}' are not supported.")

    except Exception as e:
        raise Exception(f"[ERROR] Could not load model {model_name}: {e}")

    try:
        if model_class is None:
            model = transformers.AutoModel.from_config(config)

        else:
            model = model_class(config)
    except Exception as e:
        raise Exception(f"[ERROR] Could not load model {model_name}: {e}")

    if model is None:
        raise Exception(f"Model {model_name} is not currently supported")

    available_token_classes = [
        ("Token", transformers.AutoTokenizer, {}),
        ("Image", transformers.AutoImageProcessor, {"use_fast": True}),
        ("FeatureExtractor", transformers.AutoFeatureExtractor, {}),
        ("Processor", transformers.AutoProcessor, {})
    ]

    for label, token_class, extra_args in available_token_classes:
        try:
            tokenizer = token_class.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                **{**extra_args, **kwargs}
            )
            break
        except Exception as e:
            tokenizer = None

    if tokenizer is None:
        raise Exception(f"Error initializing tokenizer for model {model_name}: {e}")

    input = None
    try:
        # Text
        if label == "Token":
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<|endoftext|>"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            text = "How to prepare coffee?"
            input = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

        # Image or Video
        elif label == "Image" or label == "FeatureExtractor" or "image" in tokenizer.__class__.__name__.lower():
            from PIL import Image
            import numpy as np

            # Check for video case based on tokenizer class name containing "video"
            if "video" in tokenizer.__class__.__name__.lower():
                # Video case: create a list of 16 frames (all white images)
                arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
                img = Image.fromarray(arr)
                video_frames = [img for _ in range(16)]
                input = tokenizer(
                    images=video_frames,
                    return_tensors="pt",
                )
            else:
                # Image case: create a single white image
                arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
                img = Image.fromarray(arr)
                input = tokenizer(
                    images=img,
                    return_tensors="pt",
                )
            input = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in input.items()}

        # Multimodal
        elif label == "Processor":
            from PIL import Image
            import numpy as np
            # Create a dummy white image
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
            img = Image.fromarray(arr)
            text = "How to prepare coffee?"
            # Combine text and image to create input for the processor
            input = tokenizer(text=text, images=img, return_tensors="pt")

    except Exception as e:
        raise Exception(f"Error creating input for model {model_name}, tokenizer {tokenizer} : {e}")

    return model, tokenizer, input


# Create tracker on different proccess
def create_tracker(log_dir, epochs, queue, ml_model=None, unsupported_models=None):
    try:
        model, tokenizer, input = load_any_model(
            ml_model.model(),
            hf_token=None,
            unsupported_models=unsupported_models,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        print(f"[CT] HF model loaded for tracker: {ml_model.model()}")

        # No-grad / eval for consistent forwards
        if hasattr(model, "eval"):
            model.eval()

        tracker = CarbonTracker(log_dir=log_dir, epochs=epochs)

        # Window of real work per epoch (seconds). Tunable via env.
        target_s = float(os.getenv("CT_TARGET_SECONDS", "1.0"))

        for epoch in range(epochs):
            tracker.epoch_start()
            start = time.time()
            iters = 0
            with torch.inference_mode():
                while (time.time() - start) < target_s:
                    try:
                        model(**input)
                    except Exception:
                        # fallback for encoder-decoder models
                        if "decoder_input_ids" not in input and "input_ids" in input:
                            input["decoder_input_ids"] = input["input_ids"]
                        model(**input)
                    iters += 1
            elapsed = time.time() - start
            tracker.epoch_end()
            print(f"[CT DEBUG][HF] epoch={epoch} iters={iters} elapsed={elapsed:.3f}s")

        tracker.stop()
        time.sleep(0.3)  # allow flush

        # One parse + one result returned
        carbon_g, energy_kwh, ci_g_per_kwh = _parse_tracker_logs_debug(log_dir)
        carbon = float(carbon_g or 0.0)
        queue.put(carbon)

    except Exception as e:
        print("[CT ERROR] create_tracker:", e)
        queue.put(e)



# Create tracker for ONNX models
def create_tracker_onnx(log_dir, epochs, queue, onnx_path):
    try:
        print(f"[CT] ONNX tracker loading: {onnx_path}")

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        if not inputs:
            raise Exception("No input tensors found in ONNX model.")
        input_name = inputs[0].name
        shape = inputs[0].shape

        # Make a deterministic dummy input (random is fine too; doesn't matter for tracking)
        dummy = np.random.rand(*[d if isinstance(d, int) else 1 for d in shape]).astype(np.float32)
        print(f"[CT] Dummy input shape: {dummy.shape}")

        tracker = CarbonTracker(log_dir=log_dir, epochs=epochs)
        target_s = float(os.getenv("CT_TARGET_SECONDS", "1.0"))

        for epoch in range(epochs):
            tracker.epoch_start()
            start = time.time()
            iters = 0
            while (time.time() - start) < target_s:
                sess.run(None, {input_name: dummy})   # <-- use 'dummy' (not dummy_input)
                iters += 1
            elapsed = time.time() - start
            tracker.epoch_end()
            print(f"[CT DEBUG][ONNX] epoch={epoch} iters={iters} elapsed={elapsed:.3f}s")

        tracker.stop()
        time.sleep(0.3)

        # One parse + one result returned
        carbon_g, energy_kwh, ci_g_per_kwh = _parse_tracker_logs_debug(log_dir)
        carbon = float(carbon_g or 0.0)
        queue.put(carbon)
    except Exception as e:
        print(f"[CT ERROR] create_tracker_onnx: {e}")
        queue.put(e)


# Signal handler
def signal_handler(sig, frame):
    print("\nExiting")
    CarbonFootprintNode.terminate()
    global running
    running = False


# User Callback implementation
# Inputs: ml_model, user_input, hw
# Outputs: node_status, co2
def task_callback(ml_model, user_input, hw, node_status, co2):

    run_tag = f"{int(time.time())}_{os.getpid()}"
    log_directory = f"/tmp/logs/carbontracker/{run_tag}"
    os.makedirs(log_directory, exist_ok=True)

    output_extra_data = {}
    # adding number of output request to extra data
    extra_data_bytes = user_input.extra_data()
    extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
    try:
        extra_data_dict = json.loads(extra_data_str)
    except json.JSONDecodeError:
        print("[WARN] In carbon node extra_data JSON is not valid.")
        extra_data_dict = {}

    # keep hardware/type from user extra data before we reuse extra_data_dict later
    user_hw_required = (extra_data_dict.get("hardware_required") or "")
    user_type = (extra_data_dict.get("type") or extra_data_dict.get("model_family") or "")

    if "num_outputs" in extra_data_dict and extra_data_dict["num_outputs"] != "":
        num_outputs = extra_data_dict["num_outputs"]
        model_restrains_list = [ml_model.model()]
        if "model_restrains" in extra_data_dict:
            model_restrains_list.extend(extra_data_dict["model_restrains"])

        output_extra_data["num_outputs"]     = num_outputs
        output_extra_data["model_restrains"] = model_restrains_list

    unsupported_models = None
    extra_data_bytes = ml_model.extra_data()
    if extra_data_bytes:
        extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
        if extra_data_str:
            try:
                extra_data_dict = json.loads(extra_data_str)
            except json.JSONDecodeError:
                print("[WARN] In ml_model node extra_data JSON is not valid.")
                extra_data_dict = {}
            if "unsupported_models" in extra_data_dict:
                unsupported_models = extra_data_dict["unsupported_models"]

    # Time to estimate Wh based on W (in hours)
    try:
        default_time = hw.latency() / (3600 * 1000)             # ms to h && W to kW
        energy_consump = hw.power_consumption()*default_time    # kW * h = kWh
    except Exception as e:
        print("Error: ", e)
        energy_consump = 0.0

    """
    # --- ONNX/FPGA SHORT-PATH: skip HF tracker and compute CO2 from HW numbers ---
    try:
        model_name = ml_model.model()
    except Exception:
        model_name = ""
    try:
        model_path = ml_model.model_path()
    except Exception:
        model_path = ""

    is_onnx = isinstance(model_path, str) and model_path.endswith(".onnx") and os.path.isfile(model_path)
    is_fpga = isinstance(user_hw_required, str) and ("fpga" in user_hw_required.lower())

    # If it's a local ONNX (our U-Net case) OR we are on FPGA, don't hit HuggingFace at all
    if is_onnx or is_fpga:
        # energy_consump is already in kWh (computed from W * h)
        energy_kwh = float(energy_consump or 0.0)

        # Choose a carbon intensity factor (grams CO2 per kWh). Adjust to your region if you want.
        carbon_intensity_g_per_kwh = 233.0  # ~EU-average

        carbon_g = energy_kwh * carbon_intensity_g_per_kwh

        # Populate outputs and return early
        co2.carbon_footprint(carbon_g)                 # grams CO2e
        co2.energy_consumption(energy_kwh)             # kWh
        co2.carbon_intensity(carbon_intensity_g_per_kwh)  # g/kWh

        # expose a bit of context for the UI/debug
        output_extra_data.update({
            "mode": "onnx_fpga_shortpath",
            "model_name": model_name,
            "model_path": model_path,
            "hardware_required": user_hw_required,
            "notes": "Used HW power/latency -> energy -> CO2 (no HF tracker)."
        })
        co2.extra_data(json.dumps(output_extra_data).encode("utf-8"))
        return
    # --- END ONNX/FPGA SHORT-PATH ---
    """
    #log_directory = "/tmp/logs/carbontracker"               # temp log dir for reading carbon data results

    # Define CarbonTracker with fallback for no available components
    try:
        queue = multiprocessing.Queue()
        ### proc = multiprocessing.Process(target=create_tracker, args=(log_directory, 1, queue, ml_model, unsupported_models))
        model_path = ml_model.model_path()
        is_onnx = isinstance(model_path, str) and model_path.endswith(".onnx")

        if is_onnx:
            print(f"[INFO] Running ONNX tracker for model: {model_path}")
            proc = multiprocessing.Process(target=create_tracker_onnx, args=(log_directory, 1, queue, model_path))
        else:
            proc = multiprocessing.Process(target=create_tracker, args=(log_directory, 1, queue, ml_model, unsupported_models))

        proc.start()
        proc.join(timeout=60)
        if proc.is_alive():
            print("Child process did not finish within the timeout period. Terminating...")
            proc.terminate()
            proc.join()
            raise Exception("tracker child process did not finish within the timeout period. Terminating...")

        if proc.exitcode == 70:
            raise Exception("No hardware components available; failed to obtain carbon footprint value.")
        else:
            if not queue.empty():
                result = queue.get()
                if isinstance(result, Exception):
                    raise Exception("Error creating tracker: " + str(result))
                else:
                    print("Tracker created successfully.")
                    carbon = float(result or 0.0)

                    # --- Prefer tracker energy if present, else fall back to HW-based estimate ---
                    tracker_energy_kwh = None
                    tracker_ci_g_per_kwh = None
                    try:
                        cg, ekwh, ci = _parse_tracker_logs_debug(log_directory)
                        # if logs contain a more precise carbon value, prefer it
                        if cg is not None and cg > 0:
                            carbon = cg
                        tracker_energy_kwh = ekwh
                        tracker_ci_g_per_kwh = ci
                    except Exception as e:
                        print("[CT DEBUG] could not re-parse tracker logs in task_callback:", e)

                    # Compute HW-based energy in kWh from latency (ms) and power (W) as a fallback
                    try:
                        latency_ms = hw.latency()
                        power_w    = hw.power_consumption()
                        # ms -> hours; W -> kW
                        default_time_h = float(latency_ms) / (3600.0 * 1000.0)
                        energy_consump_hw_kwh = (float(power_w) / 1000.0) * default_time_h
                    except Exception as e:
                        print("[CT DEBUG] HW energy compute failed:", e)
                        energy_consump_hw_kwh = 0.0

                    # Choose energy source (per-epoch/second energy)
                    if tracker_energy_kwh is not None and tracker_energy_kwh > 0:
                        energy_consump = tracker_energy_kwh
                        print(f"[CT DEBUG] Using TRACKER energy: {energy_consump:.9f} kWh")
                    else:
                        energy_consump = energy_consump_hw_kwh
                        print(f"[CT DEBUG] Using HW-estimated energy: {energy_consump:.9f} kWh (latency_ms={latency_ms}, power_w={power_w})")

                    # --- Convert epoch-based values to per-inference values (approximate) ---
                    try:
                        epoch_s = float(os.getenv("CT_TARGET_SECONDS", "1.0"))
                        latency_ms = hw.latency()
                        latency_s = float(latency_ms) / 1000.0

                        if epoch_s > 0.0 and latency_s > 0.0:
                            inf_per_epoch = epoch_s / latency_s
                            if inf_per_epoch > 0.0:
                                carbon = carbon / inf_per_epoch
                                energy_consump = energy_consump / inf_per_epoch
                                print(
                                    f"[CT DEBUG] per-inference metrics: "
                                    f"carbon_g={carbon:.9f}, energy_kwh={energy_consump:.9f}, "
                                    f"approx_inf_per_epoch={inf_per_epoch:.2f}"
                                )
                            else:
                                print("[CT DEBUG] inf_per_epoch <= 0, skipping per-inference scaling")
                        else:
                            print("[CT DEBUG] epoch_s or latency_s <= 0, skipping per-inference scaling")
                    except Exception as e:
                        print("[CT DEBUG] per-inference scaling failed:", e)

            else:
                raise Exception("No result obtained from the tracker process; failed to obtain carbon footprint value.")

        intensity = 0.0
        if energy_consump > 0:
            intensity = carbon/energy_consump

        # populate carbon footprint information
        co2.carbon_footprint(carbon)
        co2.energy_consumption(energy_consump)
        co2.carbon_intensity(intensity)
        co2.extra_data(json.dumps(output_extra_data).encode("utf-8"))

    except Exception as e:
        print(f"Error getting carbon footprint information: {e}")
        co2.carbon_footprint(0.0)
        co2.energy_consumption(0.0)
        co2.carbon_intensity(0.0)
        output_extra_data["error"] = f"Failed to obtain carbon footprint information: {e}"
        co2.extra_data(json.dumps(output_extra_data).encode("utf-8"))


# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

    # Case not supported
    res.node_id(req.node_id())
    res.transaction_id(req.transaction_id())
    error_msg = f"Unsupported configuration request: {req.configuration()}"
    res.configuration(json.dumps({"error": error_msg}))
    res.success(False)
    res.err_code(1) # 0: No error || 1: Error
    print(error_msg)


# Main workflow routine
def run():
    global running
    running = True
    node = CarbonFootprintNode(callback=task_callback, service_callback=configuration_callback)
    node.spin()


# Call main in program execution
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    """Python does not process signals async if
    the main thread is blocked (spin()) so, tun
    user work flow in another thread """
    runner = threading.Thread(target=run)
    runner.start()

    while running:
        time.sleep(1)

    runner.join()
