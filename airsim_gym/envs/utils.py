from airsim.types import Pose,Quaternionr,Vector3r
import airsim as air
import numpy as np
import time

def to_orthogonal_vectors(qr: Quaternionr):
    q0 = qr.w_val
    q1 = qr.x_val
    q2 = qr.y_val
    q3 = qr.z_val

    e1 = np.array([
        1 - 2*(q2**2 + q3**2),
        2*(q1*q2 + q0*q3),
        2*(q1*q3 - q0*q2)
    ], dtype=np.float32)

    e2 = np.array([
        2*(q1*q2 - q0*q3),
        1 - 2*(q1**2 + q3**2),
        2*(q0*q1 + q2*q3)
    ], dtype=np.float32)

    e3 = np.array([
        2*(q0*q2 + q1*q3),
        2*(q2*q3 - q0*q1),
        1 - 2*(q1**2 + q2**2)
    ], dtype=np.float32)

    return e1,e2,e3

def get_angle(v1:np.ndarray, v2:np.ndarray):
    r = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.degrees(np.arccos(r)), r

def to_quaternion(yaw, pitch, roll):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);

    q = Quaternionr(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    )

    return q

if __name__ == "__main__":
    drone = air.MultirotorClient()
    drone.reset()
    drone.enableApiControl(True)
    drone.armDisarm(True)

    drone.takeoffAsync().join()

    time.sleep(3)

    obj_pose = drone.simGetObjectPose("myobject")

    obj_crd = obj_pose.position
    obj_orn = obj_pose.orientation

    time.sleep(2)
    drone.simSetObjectPose("myobject",
        Pose(obj_crd,to_quaternion(0,45,30))
    )

    print(f"position: x={obj_crd.x_val}, y={obj_crd.y_val}, z={obj_crd.z_val}")
    print(f"orientation: w={obj_orn.w_val}, x={obj_orn.x_val}, y={obj_orn.y_val}, z={obj_orn.z_val}")
    
    e1,e2,e3 = to_orthogonal_vectors(obj_orn)

    k = drone.getMultirotorState().kinematics_estimated
    u1,u2,u3 = to_orthogonal_vectors(k.orientation)

    print(f"e1 = {e1}\ne2 = {e2}\ne3 = {e3}")
    print()
    print(f"u1 = {u1}\nu2 = {u2}\nu3 = {u3}")
    print()
    print(f"e1,u1 = {get_angle(e1,u1)}")
    print(f"e2,u2 = {get_angle(e2,u2)}")
    print(f"e3,u3 = {get_angle(e3,u3)}")

    drone.armDisarm(False)
    drone.reset()
    drone.enableApiControl(False)