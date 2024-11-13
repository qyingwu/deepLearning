import pystk
import numpy as np

def control(aim_point, current_vel, steer_factor=2.0, drift_threshold=0.49, target_vel=23):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    steer_angle = steer_factor * aim_point[0]

    # Accel control
    action.acceleration = 1.0 if current_vel < target_vel else 0.0

    # Steer control
    action.steer = np.clip(steer_angle * steer_factor, -1, 1)

    # Drift and Nitro control
    if abs(steer_angle) > drift_threshold:
        action.drift = True
        action.nitro = False
    else:
        action.drift = False
        action.nitro = True

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
