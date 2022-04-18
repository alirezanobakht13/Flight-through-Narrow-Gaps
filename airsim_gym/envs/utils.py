from airsim.types import Pose,Quaternionr,Vector3r
import numpy as np

def get_orthogonal_vectors(qr: Quaternionr):
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


if __name__ == "__main__":
    