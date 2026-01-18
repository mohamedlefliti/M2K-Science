"""
🔬 M2K SCIENTIFIC SIMULATION FRAMEWORK (FULLY FIXED + AEROSPACE EDITION v2)
Complete JAX-based framework for scientific simulations
✅ Fixed RLC circuit dynamics
✅ Fixed NameError: 'param_dimension'
✅ Added numerical stability safeguards
✅ ADDED AEROSPACE SYSTEMS:
   - AircraftLongitudinalSystem (2D, simplified)
   - RigidBodyAttitudeSystem (3D rotation)
   - FullLongitudinalFlightSystem (3D flight path with γ dynamics) ← NEW!
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable, Optional, Tuple, Union
from collections import OrderedDict

print("="*60)
print("🔬 M2K SCIENTIFIC SIMULATION FRAMEWORK (AEROSPACE EDITION v2)")
print("="*60)
print(f"✅ JAX version: {jax.__version__}")

# ============================================================================
# Core Module System
# ============================================================================

class Parameter:
    """Simple parameter container"""
    def __init__(self, data, name: str = None):
        self.data = data
        self.name = name or "param"
        self.grad = None
        self.shape = data.shape
        self.dtype = data.dtype

    def __repr__(self):
        return f"Parameter({self.shape}, {self.dtype})"

class Module:
    """PyTorch-like Module system"""

    def __init__(self, name: str = None):
        self._name = name or self.__class__.__name__
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif isinstance(value, Module):
            self._modules[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Parameter):
            self._parameters[name] = value
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self._name}.forward() not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse: bool = True):
        params = []
        for name, param in self._parameters.items():
            params.append((f"{self._name}.{name}", param))

        if recurse:
            for mod_name, module in self._modules.items():
                for param_name, param in module.parameters(recurse=True):
                    params.append((f"{self._name}.{mod_name}.{param_name}", param))

        return params

    def add_module(self, name: str, module: 'Module'):
        self._modules[name] = module
        setattr(self, name, module)

    def summary(self):
        lines = [f"Module: {self._name}"]
        total_params = 0
        for name, param in self.parameters():
            param_count = np.prod(param.shape)
            total_params += param_count
            lines.append(f"  {name}: {param.shape} ({param_count:,})")
        lines.append(f"Total parameters: {total_params:,}")
        return "\n".join(lines)

# ============================================================================
# Basic Layers
# ============================================================================

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = None):
        super().__init__(name or f"Linear_{in_features}_{out_features}")
        scale = jnp.sqrt(2.0 / in_features)
        key = jax.random.PRNGKey(42)
        weight = jax.random.normal(key, (in_features, out_features)) * scale
        self.weight = Parameter(weight, name='weight')
        if bias:
            bias_val = jnp.zeros((out_features,))
            self.bias = Parameter(bias_val, name='bias')
        else:
            self.bias = None

    def forward(self, x):
        output = jnp.dot(x, self.weight.data)
        if self.bias is not None:
            output = output + self.bias.data
        return output

class ReLU(Module):
    def forward(self, x):
        return jnp.maximum(0, x)

class Sequential(Module):
    def __init__(self, *modules, name: str = "Sequential"):
        super().__init__(name)
        self.layers = list(modules)
        for i, module in enumerate(self.layers):
            module_name = f"layer_{i}_{module.__class__.__name__}"
            module._name = module_name
            self.add_module(module_name, module)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# Scientific Simulation Infrastructure
# ============================================================================

class SimulationSystem(Module):
    """Base class for all scientific simulation systems"""

    def __init__(self, name: str = "SimulationSystem",
                 state_dim: int = 1,
                 param_dim: int = 0):
        super().__init__(name)
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.time = 0.0
        self.dt = 0.01

        self.state = Parameter(
            jnp.zeros(state_dim),
            name="state"
        )

        if self.param_dim > 0:
            self.params = Parameter(
                jnp.ones(param_dim),
                name="parameters"
            )

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement dynamics()")

    def step(self, method: str = "euler") -> jnp.ndarray:
        if method == "euler":
            dx = self.dynamics(self.state.data, self.time)
            self.state.data = self.state.data + self.dt * dx
        elif method == "rk4":
            k1 = self.dynamics(self.state.data, self.time)
            k2 = self.dynamics(self.state.data + 0.5 * self.dt * k1, self.time + 0.5 * self.dt)
            k3 = self.dynamics(self.state.data + 0.5 * self.dt * k2, self.time + 0.5 * self.dt)
            k4 = self.dynamics(self.state.data + self.dt * k3, self.time + self.dt)
            self.state.data = self.state.data + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        elif method == "rk2":
            k1 = self.dynamics(self.state.data, self.time)
            k2 = self.dynamics(self.state.data + 0.5 * self.dt * k1, self.time + 0.5 * self.dt)
            self.state.data = self.state.data + self.dt * k2

        self.time += self.dt
        return self.state.data

    def simulate(self, steps: int, method: str = "euler") -> jnp.ndarray:
        history = jnp.zeros((steps, self.state_dim))
        for i in range(steps):
            history = history.at[i].set(self.state.data)
            self.step(method)
        return history

    def reset(self, initial_state: Optional[jnp.ndarray] = None):
        self.time = 0.0
        if initial_state is not None:
            self.state.data = initial_state
        else:
            self.state.data = jnp.zeros(self.state_dim)

    def forward(self, steps: int = 100):
        return self.simulate(steps, "rk4")

# ============================================================================
# Pre-built Scientific Systems
# ============================================================================

class LorenzSystem(SimulationSystem):
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        super().__init__(name="LorenzSystem", state_dim=3, param_dim=3)
        self.params.data = jnp.array([sigma, rho, beta])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        sigma, rho, beta = params
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return jnp.array([dx, dy, dz])

class SpringMassSystem(SimulationSystem):
    def __init__(self, mass: float = 1.0, stiffness: float = 1.0, damping: float = 0.1):
        super().__init__(name="SpringMassSystem", state_dim=2, param_dim=3)
        self.params.data = jnp.array([mass, stiffness, damping])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        mass, k, c = params
        x, v = state
        dx = v
        dv = -(k/mass) * x - (c/mass) * v
        return jnp.array([dx, dv])

class LotkaVolterraSystem(SimulationSystem):
    def __init__(self, alpha: float = 1.1, beta: float = 0.4,
                 gamma: float = 0.4, delta: float = 0.1):
        super().__init__(name="LotkaVolterra", state_dim=2, param_dim=4)
        self.params.data = jnp.array([alpha, beta, gamma, delta])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        alpha, beta, gamma, delta = params
        x, y = state
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return jnp.array([dx, dy])

class ChemicalReactionSystem(SimulationSystem):
    def __init__(self, k1: float = 0.1, k2: float = 0.05, k3: float = 0.02):
        super().__init__(name="ChemicalReaction", state_dim=3, param_dim=3)
        self.params.data = jnp.array([k1, k2, k3])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        k1, k2, k3 = params
        A, B, C = state
        dA = -k1 * A + k3 * C
        dB = k1 * A - k2 * B
        dC = k2 * B - k3 * C
        return jnp.array([dA, dB, dC])

class ElectricalCircuitSystem(SimulationSystem):
    def __init__(self, R: float = 1.0, L: float = 0.5, C: float = 0.2):
        super().__init__(name="RLCCircuit", state_dim=2, param_dim=3)
        self.params.data = jnp.array([R, L, C])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        R, L, C = params
        I, V = state
        dI = -(R * I + V) / L
        dV = I / C
        return jnp.array([dI, dV])

# ============================================================================
# ✈️ AEROSPACE SYSTEMS
# ============================================================================

class AircraftLongitudinalSystem(SimulationSystem):
    def __init__(self,
                 mass: float = 1000.0,
                 wing_area: float = 20.0,
                 CD0: float = 0.02,
                 k: float = 0.04,
                 thrust: float = 5000.0,
                 gamma: float = 0.0):
        super().__init__(name="AircraftLongitudinal", state_dim=2, param_dim=6)
        self.params.data = jnp.array([mass, wing_area, CD0, k, thrust, gamma])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        m, S, CD0, k, T, gamma = params
        V, h = state
        rho0 = 1.225
        H = 8500.0
        rho = rho0 * jnp.exp(-h / H)
        q = 0.5 * rho * V**2
        D = q * S * (CD0 + k * (jnp.sin(gamma)**2))
        dV = (T - D) / m - 9.81 * jnp.sin(gamma)
        dh = V * jnp.sin(gamma)
        return jnp.array([dV, dh])

class RigidBodyAttitudeSystem(SimulationSystem):
    def __init__(self, Ixx: float = 10.0, Iyy: float = 15.0, Izz: float = 20.0):
        super().__init__(name="RigidBodyAttitude", state_dim=3, param_dim=3)
        self.params.data = jnp.array([Ixx, Iyy, Izz])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data
        Ixx, Iyy, Izz = params
        wx, wy, wz = state
        dwx = ((Iyy - Izz) / Ixx) * wy * wz
        dwy = ((Izz - Ixx) / Iyy) * wz * wx
        dwz = ((Ixx - Iyy) / Izz) * wx * wy
        return jnp.array([dwx, dwy, dwz])

# ============================================================================
# ✈️ NEW: FULL LONGITUDINAL FLIGHT DYNAMICS (FINAL TEST SYSTEM)
# ============================================================================

class FullLongitudinalFlightSystem(SimulationSystem):
    """
    Full Longitudinal Point-Mass Flight Dynamics (3-State)
    State: [V, h, gamma] → Airspeed (m/s), Altitude (m), Flight Path Angle (rad)

    Parameters (8):
      m        = mass (kg)
      S        = wing area (m²)
      CL       = lift coefficient (dimensionless)
      CD0      = zero-lift drag coefficient
      k        = induced drag factor
      thrust   = constant thrust (N)
      rho0     = sea-level air density (kg/m³, default 1.225)
      H        = atmospheric scale height (m, default 8500)
    """
    def __init__(self,
                 mass: float = 1200.0,
                 wing_area: float = 18.0,
                 CL: float = 0.5,
                 CD0: float = 0.025,
                 k: float = 0.045,
                 thrust: float = 6000.0,
                 rho0: float = 1.225,
                 H: float = 8500.0):
        super().__init__(name="FullLongitudinalFlight", state_dim=3, param_dim=8)
        self.params.data = jnp.array([mass, wing_area, CL, CD0, k, thrust, rho0, H])

    def dynamics(self, state: jnp.ndarray, t: float, params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if params is None:
            params = self.params.data

        m, S, CL, CD0, k, T, rho0, H = params
        V, h, gamma = state

        # Numerical safeguards
        V_safe = jnp.maximum(V, 1.0)
        h_clipped = jnp.maximum(h, 0.0)

        # Atmospheric model
        rho = rho0 * jnp.exp(-h_clipped / H)
        q = 0.5 * rho * V_safe**2

        # Aerodynamic forces
        L = q * S * CL
        CD = CD0 + k * CL**2
        D = q * S * CD

        g = 9.81

        # Equations of motion
        dV = (T - D) / m - g * jnp.sin(gamma)
        dh = V_safe * jnp.sin(gamma)
        dgamma = (L / (m * V_safe)) - (g * jnp.cos(gamma) / V_safe)

        return jnp.array([dV, dh, dgamma])

# ============================================================================
# Experiment Runner
# ============================================================================

class ScienceExperiment:
    def __init__(self, system: SimulationSystem):
        self.system = system
        self.results = {}
        self.config = {}

    def run(self, duration: float, dt: float = 0.01,
            method: str = "rk4", save_history: bool = True) -> Dict[str, Any]:
        steps = int(duration / dt)
        self.system.dt = dt

        if method == "euler" and dt > 0.05:
            print("⚠️  Warning: Large time step with Euler may cause instability.")

        print(f"\n🔬 Running Experiment: {self.system._name}")
        print(f"   Duration: {duration}s, Steps: {steps}, dt: {dt}")
        print(f"   Method: {method.upper()}")
        print(f"   Initial state: {self.system.state.data}")

        if hasattr(self.system, 'params'):
            print(f"   Parameters: {self.system.params.data}")

        history = self.system.simulate(steps, method)
        times = jnp.linspace(0, duration, steps)

        if jnp.any(jnp.abs(history) > 1e10):
            print("❌ Warning: Simulation shows numerical instability!")

        self.results = {
            'time': times,
            'states': history,
            'system_name': self.system._name,
            'system_type': type(self.system).__name__,
            'parameters': self.system.params.data if hasattr(self.system, 'params') else None,
            'config': {'duration': duration, 'dt': dt, 'method': method, 'steps': steps}
        }

        self._compute_statistics(history)
        print(f"✅ Simulation completed. Final state: {history[-1]}")
        return self.results

    def _compute_statistics(self, history: jnp.ndarray):
        self.results['mean'] = jnp.mean(history, axis=0)
        self.results['std'] = jnp.std(history, axis=0)
        self.results['max'] = jnp.max(history, axis=0)
        self.results['min'] = jnp.min(history, axis=0)
        self.results['final'] = history[-1]

    def plot(self, figsize=(14, 10)):
        if not self.results:
            print("❌ No results to plot.")
            return

        times = self.results['time']
        states = self.results['states']
        n_vars = states.shape[1]

        fig = plt.figure(figsize=figsize)

        ax1 = plt.subplot(2, 2, 1)
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i in range(min(n_vars, 6)):
            ax1.plot(times, states[:, i], color=colors[i % len(colors)],
                    label=f'State {i+1}', alpha=0.8, linewidth=1.5)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('State Value', fontsize=11)
        ax1.set_title(f'{self.system._name} - Time Evolution', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        if n_vars >= 2:
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(states[:, 0], states[:, 1], 'b-', alpha=0.6, linewidth=1.0)
            ax2.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
            ax2.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
            ax2.set_xlabel('State 1', fontsize=11)
            ax2.set_ylabel('State 2', fontsize=11)
            ax2.set_title('Phase Space Trajectory', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)

        if n_vars >= 3:
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax3.plot(states[:, 0], states[:, 1], states[:, 2],
                    'b-', alpha=0.6, linewidth=0.8)
            ax3.scatter(states[0, 0], states[0, 1], states[0, 2],
                       c='green', s=50, label='Start')
            ax3.scatter(states[-1, 0], states[-1, 1], states[-1, 2],
                       c='red', s=50, label='End')
            ax3.set_xlabel('X', fontsize=10)
            ax3.set_ylabel('Y', fontsize=10)
            ax3.set_zlabel('Z', fontsize=10)
            ax3.set_title('3D Phase Space', fontsize=13, fontweight='bold')
            ax3.legend(loc='best', fontsize=9)

        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"🔬 SYSTEM: {self.system._name}\n"
        stats_text += "─" * 40 + "\n"
        stats_text += f"Type: {self.results['system_type']}\n"
        stats_text += f"State dimension: {n_vars}\n"
        stats_text += f"Time steps: {len(times):,}\n"
        stats_text += f"Time span: {times[0]:.1f}s → {times[-1]:.1f}s\n"
        stats_text += f"Time step (Δt): {self.results['config']['dt']}s\n"
        stats_text += f"Method: {self.results['config']['method'].upper()}\n\n"

        if hasattr(self.system, 'params') and self.system.params.data is not None:
            stats_text += "📊 PARAMETERS:\n"
            param_names = ['σ', 'ρ', 'β'] if 'Lorenz' in self.system._name else \
                         ['m', 'k', 'c'] if 'Spring' in self.system._name else \
                         ['α', 'β', 'γ', 'δ'] if 'Lotka' in self.system._name else \
                         ['R', 'L', 'C'] if 'Circuit' in self.system._name else \
                         ['k₁', 'k₂', 'k₃'] if 'Chemical' in self.system._name else \
                         ['m', 'S', 'CD₀', 'k', 'T', 'γ'] if 'AircraftLongitudinal' in self.system._name else \
                         ['Iₓₓ', 'Iᵧᵧ', 'I_zz'] if 'RigidBody' in self.system._name else \
                         ['m', 'S', 'CL', 'CD₀', 'k', 'T', 'ρ₀', 'H'] if 'FullLongitudinal' in self.system._name else \
                         [f'p{i+1}' for i in range(len(self.system.params.data))]

            for i, (name, value) in enumerate(zip(param_names, self.system.params.data)):
                stats_text += f"  {name} = {value:.4f}\n"

        stats_text += "\n📈 STATISTICS:\n"
        for i in range(min(n_vars, 3)):
            mean_val = self.results['mean'][i]
            std_val = self.results['std'][i]
            stats_text += f"  State {i+1}: μ={mean_val:.4f}, σ={std_val:.4f}\n"

        ax4.text(0.05, 0.95, stats_text, fontfamily='monospace',
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f"Scientific Simulation Results", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return fig

# ============================================================================
# Interactive Interface
# ============================================================================

def run_interactive_simulation():
    print("\n" + "="*60)
    print("🔬 SCIENTIFIC SIMULATION WORKBENCH")
    print("="*60)

    print("\n📚 Available Systems:")
    print("1. Lorenz System")
    print("2. Spring-Mass System")
    print("3. Lotka-Volterra")
    print("4. Chemical Reaction System")
    print("5. Electrical Circuit (RLC)")
    print("6. Aircraft Longitudinal (2D)")
    print("7. Rigid Body Attitude (3D Rotation)")
    print("8. Full Longitudinal Flight (3D Path) ← FINAL TEST")

    choice = input("\n🎯 Select system (1-8): ").strip()
    system = None

    if choice == "1":
        sigma = float(input("σ (default 10.0): ") or "10.0")
        rho = float(input("ρ (default 28.0): ") or "28.0")
        beta = float(input("β (default 2.667): ") or str(8.0/3.0))
        system = LorenzSystem(sigma, rho, beta)
        x0 = float(input("x(0) (default 1.0): ") or "1.0")
        y0 = float(input("y(0) (default 1.0): ") or "1.0")
        z0 = float(input("z(0) (default 1.0): ") or "1.0")
        system.state.data = jnp.array([x0, y0, z0])

    elif choice == "2":
        mass = float(input("Mass m (default 1.0): ") or "1.0")
        stiffness = float(input("Stiffness k (default 1.0): ") or "1.0")
        damping = float(input("Damping c (default 0.1): ") or "0.1")
        system = SpringMassSystem(mass, stiffness, damping)
        x0 = float(input("x(0) (default 1.0): ") or "1.0")
        v0 = float(input("v(0) (default 0.0): ") or "0.0")
        system.state.data = jnp.array([x0, v0])

    elif choice == "3":
        alpha = float(input("α (default 1.1): ") or "1.1")
        beta = float(input("β (default 0.4): ") or "0.4")
        gamma = float(input("γ (default 0.4): ") or "0.4")
        delta = float(input("δ (default 0.1): ") or "0.1")
        system = LotkaVolterraSystem(alpha, beta, gamma, delta)
        prey0 = float(input("Prey (default 10): ") or "10.0")
        predator0 = float(input("Predator (default 5): ") or "5.0")
        system.state.data = jnp.array([prey0, predator0])

    elif choice == "4":
        k1 = float(input("k₁ (default 0.1): ") or "0.1")
        k2 = float(input("k₂ (default 0.05): ") or "0.05")
        k3 = float(input("k₃ (default 0.02): ") or "0.02")
        system = ChemicalReactionSystem(k1, k2, k3)
        A0 = float(input("[A]₀ (default 1.0): ") or "1.0")
        B0 = float(input("[B]₀ (default 0.0): ") or "0.0")
        C0 = float(input("[C]₀ (default 0.0): ") or "0.0")
        system.state.data = jnp.array([A0, B0, C0])

    elif choice == "5":
        R = float(input("R (Ω, default 1.0): ") or "1.0")
        L = float(input("L (H, default 0.5): ") or "0.5")
        C = float(input("C (F, default 0.2): ") or "0.2")
        system = ElectricalCircuitSystem(R, L, C)
        I0 = float(input("I(0) (A, default 0.0): ") or "0.0")
        V0 = float(input("V(0) (V, default 1.0): ") or "1.0")
        system.state.data = jnp.array([I0, V0])

    elif choice == "6":
        mass = float(input("Mass (kg, default 1000): ") or "1000.0")
        S = float(input("Wing area (m², default 20): ") or "20.0")
        CD0 = float(input("CD0 (default 0.02): ") or "0.02")
        k = float(input("k (default 0.04): ") or "0.04")
        thrust = float(input("Thrust (N, default 5000): ") or "5000.0")
        gamma_deg = float(input("γ (deg, default 0): ") or "0.0")
        gamma = jnp.radians(gamma_deg)
        system = AircraftLongitudinalSystem(mass, S, CD0, k, thrust, gamma)
        V0 = float(input("V(0) (m/s, default 50): ") or "50.0")
        h0 = float(input("h(0) (m, default 1000): ") or "1000.0")
        system.state.data = jnp.array([V0, h0])

    elif choice == "7":
        Ixx = float(input("Ixx (default 10): ") or "10.0")
        Iyy = float(input("Iyy (default 15): ") or "15.0")
        Izz = float(input("Izz (default 20): ") or "20.0")
        system = RigidBodyAttitudeSystem(Ixx, Iyy, Izz)
        wx0 = float(input("ωx (default 0.1): ") or "0.1")
        wy0 = float(input("ωy (default 0.05): ") or "0.05")
        wz0 = float(input("ωz (default 0.0): ") or "0.0")
        system.state.data = jnp.array([wx0, wy0, wz0])

    elif choice == "8":
        print("\n✈️ FULL Longitudinal Flight Dynamics (3-State)")
        mass = float(input("Mass (kg, default 1200): ") or "1200.0")
        S = float(input("Wing area (m², default 18): ") or "18.0")
        CL = float(input("Lift coefficient CL (default 0.5): ") or "0.5")
        CD0 = float(input("Zero-lift drag CD0 (default 0.025): ") or "0.025")
        k = float(input("Induced drag factor k (default 0.045): ") or "0.045")
        thrust = float(input("Thrust (N, default 6000): ") or "6000.0")
        system = FullLongitudinalFlightSystem(mass, S, CL, CD0, k, thrust)
        V0 = float(input("Initial airspeed V (m/s, default 60): ") or "60.0")
        h0 = float(input("Initial altitude h (m, default 2000): ") or "2000.0")
        gamma_deg = float(input("Initial flight path angle γ (deg, default 5): ") or "5.0")
        gamma0 = jnp.radians(gamma_deg)
        system.state.data = jnp.array([V0, h0, gamma0])

    else:
        print("⚠️ Invalid choice. Using default Lorenz system.")
        system = LorenzSystem()
        system.state.data = jnp.array([1.0, 1.0, 1.0])

    duration = float(input("Duration (s, default 50): ") or "50")
    dt = float(input("Time step Δt (default 0.01): ") or "0.01")
    method = input("Method (euler/rk2/rk4, default rk4): ").lower() or "rk4"

    print("\n" + "="*60)
    experiment = ScienceExperiment(system)
    results = experiment.run(duration, dt, method)
    experiment.plot()

    save = input("\n💾 Save results? [y/N]: ").lower()
    if save == 'y':
        filename = input("Filename (default: simulation): ") or "simulation"
        np.save(f"{filename}.npy", results)
        with open(f"{filename}_summary.txt", 'w') as f:
            f.write(f"M2K Scientific Simulation Results\n")
            f.write(f"System: {results['system_name']}\n")
            f.write(f"Final state: {results['final']}\n")
        print(f"✅ Saved to {filename}.npy and {filename}_summary.txt")

    print("\n✅ EXPERIMENT COMPLETED!")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 M2K SCIENTIFIC SIMULATION FRAMEWORK (AEROSPACE EDITION v2)")
    print("="*60)
    print("\n📋 Menu Options:")
    print("1. 🎮 Interactive simulation")
    print("2. 🧪 Batch experiments")
    print("3. 🚀 Quick demo (all systems, including aerospace)")

    option = input("\nSelect option (1-3): ").strip()

    try:
        if option == "1":
            run_interactive_simulation()
        elif option == "2":
            print("Batch mode not fully implemented — running quick demo instead.")
            option = "3"
        if option == "3":
            print("\n🚀 QUICK DEMO: TESTING ALL SYSTEMS (INCLUDING FINAL AEROSPACE TEST!)")
            systems = [
                ("Lorenz", LorenzSystem(), jnp.array([1.0, 1.0, 1.0]), 20.0),
                ("Spring-Mass", SpringMassSystem(), jnp.array([1.0, 0.0]), 10.0),
                ("Lotka-Volterra", LotkaVolterraSystem(), jnp.array([10.0, 5.0]), 50.0),
                ("Chemical", ChemicalReactionSystem(), jnp.array([1.0, 0.0, 0.0]), 100.0),
                ("RLC Circuit", ElectricalCircuitSystem(), jnp.array([0.0, 1.0]), 30.0),
                ("Aircraft Long.", AircraftLongitudinalSystem(), jnp.array([50.0, 1000.0]), 100.0),
                ("Rigid Body", RigidBodyAttitudeSystem(), jnp.array([0.1, 0.05, 0.0]), 50.0),
                ("Full Flight", FullLongitudinalFlightSystem(), jnp.array([60.0, 2000.0, jnp.radians(5.0)]), 120.0)
            ]

            for name, system, init_state, duration in systems:
                print(f"\n▶️ Testing {name}...")
                system.state.data = init_state
                exp = ScienceExperiment(system)
                exp.run(duration=duration, dt=0.01, method="rk4")
                plt.figure(figsize=(6, 4))
                plt.plot(exp.results['time'], exp.results['states'][:, 0], label='Primary variable')
                plt.title(f"{name}")
                plt.xlabel('Time')
                plt.ylabel('State')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()

            print("\n✅ All systems ran successfully — including the FINAL AEROSPACE TEST!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "="*60)
        print("🎉 M2K Framework — Ready for Master's Thesis in Scientific Computing!")
        print("="*60)
