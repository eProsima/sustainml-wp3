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
import os
import signal
import threading
import time
import torch
import transformers

# Whether to go on spinning or interrupt
running = False


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
        print(f"Model {ml_model.model()} loaded successfully.")
        # Define CarbonTracker
        tracker = CarbonTracker(log_dir=log_dir, epochs=epochs)
        for epoch in range(epochs):
            # Start measuring
            tracker.epoch_start()
            # Execute the training task
            # ...
            try:
                model(**input)
            except Exception:
                if "decoder_input_ids" not in input and "input_ids" in input:
                    input["decoder_input_ids"] = input["input_ids"]
                try:
                    model(**input)
                except Exception as e_model2:
                    raise Exception(e_model2)

            # Stop measuring
            tracker.epoch_end()
        tracker.stop()

        # Retrieve carbon information
        try:
            logs = parser.parse_all_logs(log_dir=log_dir)
        except Exception as e:
            print("Error: ", e)
            logs = None
        if logs:
            for entry in reversed(logs):
                pred = entry.get("pred")
                if pred and pred.get("co2eq (g)", 0) > 0:
                    carbon = pred.get("co2eq (g)", 0)
                    break
            else:
                carbon = 0.0
                raise RuntimeError("No non-zero CarbonTracker entry found")
        else:
            carbon = 0.0

        queue.put(carbon)
    except Exception as e:
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
        carbon_intensity_g_per_kwh = 233.0  # ~EU-average example

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


    log_directory = "/tmp/logs/carbontracker"               # temp log dir for reading carbon data results

    # Define CarbonTracker with fallback for no available components
    try:
        queue = multiprocessing.Queue()
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
                    carbon = result
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
