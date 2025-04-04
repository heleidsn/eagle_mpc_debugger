import dynamixel_sdk as dxl  # Import Dynamixel SDK

# Define serial port and baud rate (modify if needed)
DEVICENAME = "/dev/ttyUSB0"   # Check using `ls /dev/ttyUSB*`
BAUDRATE = 57600             # Try 1000000 if this doesn't work
PROTOCOL_VERSION = 2.0       # Change to 1.0 for older motors

# Create port and packet handlers
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print(f"Opened port: {DEVICENAME}")
else:
    print(f"Failed to open port: {DEVICENAME}")
    exit()

# Set baud rate
if portHandler.setBaudRate(BAUDRATE):
    print(f"Baudrate set to: {BAUDRATE}")
else:
    print("Failed to set baudrate")
    exit()

# Scan for Dynamixel IDs
print("Scanning for Dynamixel motors...")
for i in range(0, 253):  # Dynamixel ID range: 0–252
    dxl_model, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, i)
    if dxl_comm_result == dxl.COMM_SUCCESS:
        print(f"✅ Found Dynamixel Motor! ID: {i}, Model: {dxl_model}")

# Close port
portHandler.closePort()
