import asyncio
from bleak import BleakClient

# DX-BT24 MAC Address
DEVICE_ADDRESS = "48:87:2D:69:6A:6C"

# DX-BT24 Write Characteristic UUID
WRITE_CHARACTERISTIC_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"

async def main():
    print(f"Connecting to {DEVICE_ADDRESS}...")
    
    # Connect ONCE here
    async with BleakClient(DEVICE_ADDRESS) as client:
        if client.is_connected:
            print("Connected successfully!")
            print("Controls: Type '1' (ON), '0' (OFF), or 'q' (Quit)\n")

            # Continuous loop over the SAME active connection
            while True:
                # Prompt for input without blocking event loop
                user_input = await asyncio.to_thread(input, "Enter command (1/0/q): ")
                user_input = user_input.strip()

                if user_input.lower() == 'q':
                    print("Exiting program and disconnecting...")
                    break  # Exiting the loop closes the connection block cleanly
                
                if user_input in ['0', '1']:
                    # Send signal over the existing connection
                    await client.write_gatt_char(
                        WRITE_CHARACTERISTIC_UUID, 
                        user_input.encode('utf-8')
                    )
                    print(f"--> Sent '{user_input}' over active connection.")
                else:
                    print("Invalid option. Type '1', '0', or 'q'.")

if __name__ == "__main__":
    asyncio.run(main())