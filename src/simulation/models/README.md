# MuJoCo Model Files

This directory contains MuJoCo XML model definitions for the robotic arm and environments.

## File Organization

- `arm_v1.xml` - Initial robotic arm model definition
- `arm_v2.xml` - Improved arm model (as design evolves)
- `objects/` - Object models for manipulation (spheres, boxes, etc.)
- `scenes/` - Complete scene setups (arm + objects + environment)
- `meshes/` - STL/OBJ mesh files if using complex geometries

## Creating MuJoCo Models

### Basic Structure

Each MuJoCo XML file should include:

1. **Compiler settings** - Units, coordinate frame conventions
2. **Options** - Solver settings, timestep, gravity
3. **Assets** - Textures, meshes, materials
4. **Worldbody** - The physical scene (arm, objects, floor, etc.)
5. **Actuators** - Motors/controllers for the joints
6. **Sensors** - Position, velocity, force sensors

### Robotic Arm Model Requirements

The arm model should include:

- **Kinematic chain** - Series of bodies connected by joints
  - Base (attached at waist position)
  - Upper arm segment
  - Forearm segment
  - Wrist
  - Gripper/end-effector

- **Joints** - DOF for each segment
  - Shoulder: 3 DOF (spherical or 3 revolute)
  - Elbow: 1-2 DOF
  - Wrist: 2-3 DOF
  - Gripper: 1-2 DOF for grasping

- **Actuators** - One per DOF
  - Motor type (position, velocity, or torque control)
  - Force/torque limits
  - Gear ratios

- **Collision geometries** - For contact detection
  - Simple shapes (cylinders, boxes) for efficiency
  - Separate visual and collision meshes if needed

- **Sensors** - For observation space
  - Joint position sensors
  - Joint velocity sensors
  - Touch sensors on gripper

### Example XML Structure (pseudocode)

```xml
<mujoco model="reach_arm">
  <compiler angle="radian" .../>
  
  <option timestep="0.002" .../>
  
  <worldbody>
    <body name="base" pos="0 0 1">  <!-- waist height -->
      <geom type="cylinder" .../>
      
      <body name="upper_arm">
        <joint name="shoulder_flex" .../>
        <geom type="capsule" .../>
        
        <body name="forearm">
          <joint name="elbow_flex" .../>
          <geom type="capsule" .../>
          
          <!-- Continue kinematic chain... -->
        </body>
      </body>
    </body>
    
    <!-- Floor, walls, objects, etc. -->
  </worldbody>
  
  <actuator>
    <motor joint="shoulder_flex" .../>
    <motor joint="elbow_flex" .../>
    <!-- One actuator per joint -->
  </actuator>
  
  <sensor>
    <jointpos joint="shoulder_flex" .../>
    <jointvel joint="shoulder_flex" .../>
    <!-- Sensors for each joint -->
  </sensor>
</mujoco>
```

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [dm_control Examples](https://github.com/deepmind/dm_control)

## Design Notes

TODO: Document design decisions here as the arm model evolves:
- Joint limits chosen based on...
- Actuator strengths based on...
- Collision geometry simplifications...
- etc.

