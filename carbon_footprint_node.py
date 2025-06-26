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
import signal
import threading
import time
import json
import multiprocessing
import transformers
import torch

# Whether to go on spinning or interrupt
running = False

def load_any_model(model_name, hf_token=None, **kwargs):

    model = None

    try:
        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_class = transformers.AutoModel._model_mapping.get(type(config), None)

        if "llama" in model_class.__name__.lower() or \
        "mistral" in model_class.__name__.lower() or \
        "qwen" in model_class.__name__.lower() or \
        "phi3" in model_class.__name__.lower() or \
        "t5" in model_class.__name__.lower():
            raise ValueError("Models that use 'llama', 'mistral', 'qwen', 'phi3' or 't5' are not supported.")
    except Exception as e:
        raise Exception(f"[ERROR] Could not load model {model_name} configuration: {e}")

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
def create_tracker(log_dir, epochs, queue, ml_model=None):
    try:
        model, tokenizer, input = load_any_model(
            ml_model.model(),
            hf_token=None,
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
                output = model(**input)
            except Exception as e_model:
                if "decoder_input_ids" not in input and "input_ids" in input:
                    input["decoder_input_ids"] = input["input_ids"]
                try:
                    output = model(**input)
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
    extra_data_dict = json.loads(extra_data_str)

    if "num_outputs" in extra_data_dict and extra_data_dict["num_outputs"] != "":
        num_outputs = extra_data_dict["num_outputs"]
        model_restrains_list = [ml_model.model()]
        if "model_restrains" in extra_data_dict:
            model_restrains_list.extend(extra_data_dict["model_restrains"])

        output_extra_data["num_outputs"]     = num_outputs
        output_extra_data["model_restrains"] = model_restrains_list

    # Time to estimate Wh based on W (in hours)
    try:
        default_time = hw.latency() / (3600 * 1000)             # ms to h && W to kW
        energy_consump = hw.power_consumption()*default_time    # kW * h = kWh
    except Exception as e:
        print("Error: ", e)
        energy_consump = 0.0

    log_directory = "/tmp/logs/carbontracker"               # temp log dir for reading carbon data results

    # Define CarbonTracker with fallback for no available components
    try:
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=create_tracker, args=(log_directory, 1, queue, ml_model))
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
