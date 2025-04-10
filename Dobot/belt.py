import DobotDllType as dType
import time

# Connect to the Dobot Magician
CON_STR = 'COM4'  # Change this to match your Dobot Magician's serial port
api = dType.load()


# Initialize the Dobot Magician
state = dType.ConnectDobot(api, "COM4", 115200)
if state[0] != 0:
    print("Failed to connect to Dobot Magician")
    exit()

# Define the photoelectric sensor pin
sensor_pin = 2

# Move the conveyer belt forward at a constant speed
speed = 200  # Adjust this to match your conveyer belt's speed
dType.SetEMotor(api, 0, 1, speed, 0)

# Read the sensor data and stop the belt if an object is detected
while True:
    # Read the sensor data
    sensor_value = dType.GetIODI(api, sensor_pin)[0]
    print(sensor_value)
    # Stop the belt if an object is detected
    if sensor_value == 1:
        dType.SetEMotor(api, 0, 1, 0, 0)
        break

    # Delay for a short period to prevent excessive loop iterations
    time.sleep(0.1)

# Disconnect from the Dobot Magician
dType.DisconnectDobot(api)
