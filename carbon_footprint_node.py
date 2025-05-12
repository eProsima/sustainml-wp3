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

# Whether to go on spinning or interrupt
running = False

# Create tracker on different proccess
def create_tracker(log_dir, epochs, queue):
    try:
        # Define CarbonTracker
        tracker = CarbonTracker(log_dir=log_dir, epochs=epochs)
        for epoch in range(epochs):
            # Start measuring
            tracker.epoch_start()
            # Execute the training task
            # ...
            time.sleep(5)   # 5 seconds sleep as training (temporal approach) TODO
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
        proc = multiprocessing.Process(target=create_tracker, args=(log_directory, 1, queue))
        proc.start()
        proc.join(timeout=10)
        if proc.is_alive():
            print("Child process did not finish within the timeout period. Terminating...")
            proc.terminate()
            proc.join()
            carbon = 0.0

        if proc.exitcode == 70:
            print("Warning: No hardware components available, using dummy carbon value.")
            carbon = 0.0
        else:
            if not queue.empty():
                result = queue.get()
                if isinstance(result, Exception):
                    print("Error creating tracker:", result)
                    carbon = 0.0
                else:
                    print("Tracker created successfully.")
                    carbon = result
            else:
                carbon = 0.0

        intensity = 0.0
        if energy_consump > 0:
            intensity = carbon/energy_consump

        # populate carbon footprint information
        co2.carbon_footprint(carbon)
        co2.energy_consumption(energy_consump)
        co2.carbon_intensity(intensity)

        # adding number of output request to extra data
        extra_data_bytes = user_input.extra_data()
        extra_data_str = ''.join(chr(b) for b in extra_data_bytes)
        extra_data_dict = json.loads(extra_data_str)

        if "num_outputs" in extra_data_dict and extra_data_dict["num_outputs"] != "":
            num_outputs = extra_data_dict["num_outputs"]
            print(f"Number of outputs: {num_outputs}")  #debug
            model_restrains_list = [ml_model.model()]
            if "model_restrains" in extra_data_dict:
                model_restrains_list.extend(extra_data_dict["model_restrains"])

            encoded_data = json.dumps({"num_outputs": num_outputs, "model_restrains": model_restrains_list}).encode("utf-8")
            co2.extra_data(encoded_data)

    except Exception as e:
            print(f"Error getting carbon footprint information: {e}")

# User Configuration Callback implementation
# Inputs: req
# Outputs: res
def configuration_callback(req, res):

    # Callback for configuration implementation here

    # Dummy JSON configuration and implementation
    dummy_config = {
        "param1": "value1",
        "param2": "value2",
        "param3": "value3"
    }
    res.configuration(json.dumps(dummy_config))
    res.node_id(req.node_id())
    res.transaction_id(req.transaction_id())
    res.success(True)
    res.err_code(0) # 0: No error || 1: Error

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
