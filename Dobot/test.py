'''

Sample code to move to point A then point B 

'''

import DoBotArm as Dbt
import DobotDllType as dType
 
def main():
    #api object to access the API functions
    api = dType.load()
    #establish connetion with one dobot on specific COM port
    state = dType.ConnectDobot(api, "COM7", 115200)
    if state == 0:
        print("Successfully connected to the Dobot Magician robot")
    else:
        print("Failed to connect to the Dobot Magician robot")
        exit()

    #dobot working parameters for point to point movement dType.SetPTPCommonParams(api, velocity, acceleration, isQueued)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)

    # Set the target coordinates for point A and point B
    point_A = [200, 0, 50, 0]
    point_B = [200, 100, 50, 0]

    # Move the robot arm to point A
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, *point_A, isQueued=1)

    # Wait for the robot arm to finish moving
    while True:
        if dType.GetQueuedCmdCurrentIndex(api, None)[0] == -1:
            break

    # Move the robot arm to point B
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, *point_B, isQueued=1)

    # Wait for the robot arm to finish moving
    while True:
        if dType.GetQueuedCmdCurrentIndex(api, None)[0] == -1:
            break

    # Disconnect from the Dobot Magician robot
    dType.DisconnectDobot(api)


main()