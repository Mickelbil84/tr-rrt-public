#####################################################
# Motion planning module, for abstract MP algorithms
#####################################################

class MotionPlannerSettings(object):
    def __init__(self):
        pass


class MotionPlanner(object):
    def __init__(self, settings):
        self.settings = settings

    def plan(self, q1: object, q2: object):
        """
        Plan a path from configuration q1 to configuration q2.
        A configuration can be any type the planner can handle.

        Returns: A list of configurations, representing a path from q1 to q2
        """
        raise NotImplementedError()