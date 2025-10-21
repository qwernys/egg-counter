from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock

from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian

import threading
import argparse

import os

import time

def modbus_server(context):
    print(f"Starting Modbus server at 0.0.0.0:5020")
    StartTcpServer(context, address=("0.0.0.0", 5020))

def get_register(value):
    """Convert a 32-bit integer to Modbus holding register format."""

    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_uint(value)
    return builder.to_registers()

def parse_args():
    parser = argparse.ArgumentParser(description="Egg Counter")

    parser.add_argument("--data-dir", type=str, default="/app/data", 
                        help="Directory to store data files")

    return parser.parse_args()

def get_total_count(data_dir):
    total_count = 0
    for cameraId in range(1,5):
        count_a_path = os.path.join(data_dir, f"count_{cameraId}a.txt")
        count_b_path = os.path.join(data_dir, f"count_{cameraId}b.txt")

        with open(count_a_path, "r") as f:
            total_count += int(f.read().strip())
        with open(count_b_path, "r") as f:
            total_count += int(f.read().strip())
    return total_count

def get_daily_count(data_dir):
    daily_count = 0
    for cameraId in range(1,5):
        daily_a_path = os.path.join(data_dir, f"daily_{cameraId}a.txt")
        daily_b_path = os.path.join(data_dir, f"daily_{cameraId}b.txt")

        with open(daily_a_path, "r") as f:
            daily_data = f.read().strip()
            _, count_str = daily_data.split(',')
            daily_count += int(count_str)
        with open(daily_b_path, "r") as f:
            daily_data = f.read().strip()
            _, count_str = daily_data.split(',')
            daily_count += int(count_str)
    return daily_count    

def main(args):
    # Modbus context setup
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0]*34)  # 34 holding registers
    )
    context = ModbusServerContext(slaves=store, single=True)
    ### Fix Init? ### 
    
    #context[0].setValues(3, 0, get_register(total_count))  # Initialize register 0 with total count
    #context[0].setValues(3, 2, get_register(daily_count))  # Initialize register 1 with daily count
    #context[0].setValues(3, 4 + (cameraId - 1) * 8, get_register(total_count_xa)) # Initialize register 14 with total count for 4a
    #context[0].setValues(3, 6 + (cameraId - 1) * 8, get_register(daily_xa))  # Initialize register 15 with daily count for 4a
    #context[0].setValues(3, 8 + (cameraId - 1) * 8, get_register(total_count_xb)) # Initialize register 16 with total count for 4b
    #context[0].setValues(3, 10 + (cameraId - 1) * 8, get_register(daily_xb))  # Initialize register 17 with daily count for 4b

    threading.Thread(target=modbus_server, args=(context,), daemon=True).start()


    while True:
        total_count = get_total_count(args.data_dir)
        daily_count = get_daily_count(args.data_dir)

        context[0].setValues(3, 0, get_register(total_count))  # Update total count register
        context[0].setValues(3, 2, get_register(daily_count))  # Update daily count register

        for cameraId in range(1,5):
            count_xb_path = os.path.join(args.data_dir, f"count_{cameraId}b.txt")
            count_xa_path = os.path.join(args.data_dir, f"count_{cameraId}a.txt")

            daily_xa_path = os.path.join(args.data_dir, f"daily_{cameraId}a.txt")
            daily_xb_path = os.path.join(args.data_dir, f"daily_{cameraId}b.txt")

            with open(count_xb_path, "r") as f:
                total_count_xb = int(f.read().strip())

            with open(count_xa_path, "r") as f:
                total_count_xa = int(f.read().strip())

            with open(daily_xa_path, "r") as f:
                daily_data = f.read().strip()
                _, count_str = daily_data.split(',')
                total_daily_xa = int(count_str)

            with open(daily_xb_path, "r") as f:
                daily_data = f.read().strip()
                _, count_str = daily_data.split(',')
                total_daily_xb = int(count_str)

            context[0].setValues(3, 4 + (cameraId - 1) * 8, get_register(total_count_xa))
            context[0].setValues(3, 6 + (cameraId - 1) * 8, get_register(total_daily_xa))
            context[0].setValues(3, 8 + (cameraId - 1) * 8, get_register(total_count_xb))
            context[0].setValues(3, 10 + (cameraId - 1) * 8, get_register(total_daily_xb))

        time.sleep(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)