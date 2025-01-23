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

# Whether to go on spinning or interrupt
running = False

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
    default_time = hw.latency() / (3600 * 1000)             # ms to h
    energy_consump = hw.power_consumption()*default_time    # W * h = Wh
    log_directory = "/tmp/logs/carbontracker"               # temp log dir for reading carbon data results

    # Define CarbonTracker
    tracker = CarbonTracker(log_dir=log_directory, epochs=1)
    # Start measuring
    tracker.epoch_start()
    # Execute the training task
    # ...
    time.sleep(2)   # 2 seconds sleep as training (temporal approach)
    # Stop measuring
    tracker.epoch_end()
    tracker.stop()

    # Retrieve carbon information
    logs = parser.parse_all_logs(log_dir=log_directory)
    first_log = logs[0]
    carbon = first_log['pred']['co2eq (g)']
    intensity = 0.0
    if energy_consump > 0:
        intensity = carbon/energy_consump

    # populate carbon footprint information
    co2.carbon_footprint(carbon)
    co2.energy_consumption(energy_consump)
    co2.carbon_intensity(intensity)

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
