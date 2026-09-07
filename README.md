# Jacobian

A small Python project demonstrating **Jacobian-based inverse kinematics** for robotic arms.

The project includes 2D and 3D simulations that use the Jacobian to determine how joint angles should change to move an end effector toward a target.

## Setup

Clone the repository:

```bash
git clone https://github.com/dinithwijeratne/jacobian.git
cd jacobian
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

## Usage

### 2D Simulation

Run:

```bash
python main2d.py
```

This demonstrates Jacobian-based control of a 2-link planar robotic arm.

### 3D Simulation

Run:

```bash
python main3d.py
```

This demonstrates Jacobian-based inverse kinematics for a simple 3D robotic arm.

### Interactive Controller

Run:

```bash
python mainControl.py
```

This allows the target position to be moved interactively while the robot attempts to follow it.

## Use Case

The project is intended as a practical exploration of **robotics, linear algebra, and control theory**, specifically how the Jacobian can be used to convert desired end-effector motion into changes in robot joint angles.

It serves as a foundation for more advanced robotic control and inverse-kinematics implementations.
