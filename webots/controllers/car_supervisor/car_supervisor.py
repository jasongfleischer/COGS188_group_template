from controller import Supervisor


START_COORDS = [56, -47.75, 0.316922]
START_ROTATION = [0.0, -1.0, 0.0, 0.0]

VIEW_START = [23.1356, -48.1635, 22.6393]

class CarSupervisor:
    def __init__(self):
        self.supervisor = Supervisor()
        
        self.car_node = self.supervisor.getFromDef("CAR")
        self.viewpoint_node = self.supervisor.getFromDef("VIEW")
        
        self.translation_field = self.car_node.getField("translation")
        self.rotation_field = self.car_node.getField("rotation")
        self.time_step = int(self.supervisor.getBasicTimeStep())
        
        self.view_pos = self.viewpoint_node.getField("position")
        
        self.reset_flag = self.supervisor.getFromDef("RESET_FLAG")

    def reset_car_position(self):
        """ Resets the car's position and rotation """
        self.translation_field.setSFVec3f(START_COORDS)
        self.rotation_field.setSFRotation(START_ROTATION)
        self.supervisor.simulationResetPhysics()
        
        self.view_pos.setSFVec3f(VIEW_START)
        
        
        self.reset_flag.getField("translation").setSFVec3f([0, 0, 0])

    def run(self):
        while self.supervisor.step(self.time_step) != -1:
            reset_pos = self.reset_flag.getField("translation").getSFVec3f()
            if reset_pos[0] > 0: 
                self.reset_car_position()
                
if __name__ == "__main__":
    supervisor = CarSupervisor()
    supervisor.run()