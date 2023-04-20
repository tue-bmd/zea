"""Control module for ultrasound image processing"""

class PIDController:
    """PID controller for ultrasound image processing"""
    def __init__(self, Kp, Ki, Kd, setpoint, min_val, max_val, metric=None):
        """Initialize the PID controller

        args:
            Kp (float): Proportional gain
            Ki (float): Integral gain
            Kd (float): Derivative gain
            setpoint (float): Setpoint of the controller
            min_val (float): Minimum output value
            max_val (float): Maximum output value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.min_val = min_val
        self.max_val = max_val

        self.setpoint = setpoint
        self.dt = 1

        self.error = 0
        self.error_last = 0
        self.error_sum = 0
        self.error_diff = 0

        self.output = 0

        if metric:
            self.metric = metric
        else:
            self.metric = lambda x: x

    def update(self, x):
        """Update the PID controller with a new image"""

        error = self.setpoint - self.metric(x)

        self.error = error
        self.error_sum += self.error
        self.error_diff = (self.error - self.error_last)

        self.error_last = self.error

        self.output = self.Kp * self.error + self.Ki * \
            self.error_sum + self.Kd * self.error_diff

        return self.output

    def reset(self):
        """Reset the PID controller"""
        self.error = 0
        self.error_last = 0
        self.error_sum = 0
        self.error_diff = 0
        self.output = 0
