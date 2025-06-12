import dynamixel_sdk as dxl

# 设备端口（Linux一般是 /dev/ttyUSB0）
DEVICENAME = "/dev/ttyUSB0"

# 波特率（Dynamixel 默认是 57600）
BAUDRATE = 57600

# 使用 PacketHandler 和 PortHandler
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(2.0)  # 2.0 代表 Dynamixel 协议版本

# 打开端口
if portHandler.openPort():
    print("端口打开成功")
else:
    print("端口打开失败")

# 设置波特率
if portHandler.setBaudRate(BAUDRATE):
    print("波特率设置成功")
else:
    print("波特率设置失败")
