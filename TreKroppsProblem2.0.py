import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve  # Added for Euler backward

# Definerar konstanter: massor positioner och hastigheter i np arrayer, vektorer
G = 1.0  # G som 1 hör att göra systemet dimentionslöst

m1 = 1.0
m2 = 1.0
m3 = 1.0
m = np.array([m1, m2, m3])



# Instabil Fjärilsbana
r1 = np.array([-1.0, 0.0, 0.0])
r2 = np.array([1, 0.0, 0.0])
r3 = np.array([0.0, 0.0, 0.0])
R = np.vstack([r1, r2, r3])

omega = np.sqrt(1.0)  # VInkelhastighet, använder inte
v1 = np.array([0.405916, 0.230163, 0.0])
v2 = np.array([0.405916, 0.230163, 0.0])
v3 = np.array([-0.811832, -0.460326, 0.0])




'''

#Stabil cirkulär bana
r1 = np.array([ 1/np.sqrt(3),  0.0, 0.0])
r2 = np.array([-0.5/np.sqrt(3),  0.5, 0.0])
r3 = np.array([-0.5/np.sqrt(3), -0.5, 0.0])
R = np.vstack([r1, r2, r3])

v1 = np.array([0.0,  1.0, 0.0])
v2 = np.array([-np.sqrt(3)/2, -0.5, 0.0])
v3 = np.array([ np.sqrt(3)/2, -0.5, 0.0])
'''





y0 = np.hstack([r1, v1, r2, v2, r3, v3])  # Definierar en start statevector


def pairwise_accel(ri, rj, mj, G=1.0, eps=0):
    """
    Accelerationen på en kropp vid ri på grund av en kropp mj på avstånd rj
    """
    dr = rj - ri # Definerar skillnaden mellan rj-ri
    dist2 = np.dot(dr, dr) + eps**2 # Storleken av dr
    inv_dist3 = dist2**(-1.5) # Definerar nämnaren
    return G * mj * dr * inv_dist3 # Utför beräkningen


def accelerations(R, m, G=1.0, eps=0):
    """
    Beräknar parvisa accelerationer av alla kroppar på alla kroppar
    """
    a = np.zeros_like(R) # Definierar tom vektor av samma storlekn som R
    # på kropp 1 från 2 och 3
    a[0] = pairwise_accel(R[0], R[1], m[1], G, eps) + pairwise_accel(R[0], R[2], m[2], G, eps)
    # på kropp 2 från 1 och 3
    a[1] = pairwise_accel(R[1], R[0], m[0], G, eps) + pairwise_accel(R[1], R[2], m[2], G, eps)
    # på kropp 3 från 1 och 2
    a[2] = pairwise_accel(R[2], R[0], m[0], G, eps) + pairwise_accel(R[2], R[1], m[1], G, eps)
    return a

a = accelerations(R, m, G=G, eps=1e-3) # 3x1 kolumnvektor bestående av tre radvektorer, bestående av den totala momentana tredimentionella accelerationen hos alla separata kroppar.


def deriv(t, y, m, G=1.0):
    """
    Beräknar tidsderivatan för statevector y
    """
    dydt = np.zeros_like(y) # kapar tom vektor i samma dimentioner som y

    # Extrahera position och hastigheter från vår statevector y
    r1, v1 = y[0:3], y[3:6]
    r2, v2 = y[6:9], y[9:12]
    r3, v3 = y[12:15], y[15:18]

    R = np.vstack([r1, r2, r3])
    a = accelerations(R, m, G)

    # Assignar derivator
    dydt[0:3] = v1
    dydt[3:6] = a[0]
    dydt[6:9] = v2
    dydt[9:12] = a[1]
    dydt[12:15] = v3
    dydt[15:18] = a[2]

    return dydt




# Tidssteg Storlek
h = 0.01  
t0 = 0.0  # starttid = 0
t_max = 100.0      # total simuleringstid i simulation units [su]
steps = int(t_max / h) # Summan av totala states i vår burade simulering






'''
# Steg med Euler forward
def euler_forward_step(func, t, y, h, *args):
    """
    Euler forward steg
    """
    return y + h * func(t, y, *args)

# Initiera arrayer
Y = np.zeros((steps, 18))  # Vektor av vektorer av samtliga av simuleringens tillstånd
T = np.zeros(steps)        # tidpunkter

# Startvillkor
Y[0] = y0
T[0] = t0

# Loopa
for i in range(1, steps):
    Y[i] = euler_forward_step(deriv, T[i-1], Y[i-1], h, m, G)
    T[i] = T[i-1] + h
'''






'''
# Steg med Euler backward
def euler_backward_step(func, t, y, h, *args):
    """
    Euler backward steg
    """
    def residual(y_next):
        return y_next - y - h * func(t + h, y_next, *args)
    
    y_next_guess = y + h * func(t, y, *args)  # Use explicit Euler as initial guess
    y_next = fsolve(residual, y_next_guess)
    return y_next

# Initiera arrayer
Y = np.zeros((steps, 18))  # Vektor av vektorer av samtliga av simuleringens tillstånd
T = np.zeros(steps)        # tidpunkter

# Startvillkor
Y[0] = y0
T[0] = t0

# Loopa
for i in range(1, steps):
    Y[i] = euler_backward_step(deriv, T[i-1], Y[i-1], h, m, G)
    T[i] = T[i-1] + h
'''







# Steg med egenskriven RK4
def rk4_step(func, t, y, h, *args):
    """
    RK4 steg
    """
    k1 = func(t, y, *args)
    k2 = func(t + 0.5*h, y + 0.5*h*k1, *args)
    k3 = func(t + 0.5*h, y + 0.5*h*k2, *args)
    k4 = func(t + h, y + h*k3, *args)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Initiera arrayer
Y = np.zeros((steps, 18))  # Vektor av vektorer av samtliga av simuleringens tillstånd
T = np.zeros(steps)        # tidpunkter

# Startvillkor f
Y[0] = y0
T[0] = t0

# Loopa
for i in range(1, steps):
    Y[i] = rk4_step(deriv, T[i-1], Y[i-1], h, m, G)
    T[i] = T[i-1] + h







'''
# Steg med inbygda scipy.integrates RK45
sol = solve_ivp(fun=lambda t, y: deriv(t, y, m, G),
                t_span=(t0, t_max),
                y0=y0,
                method='RK45',
                rtol=1e-9, atol=1e-10,   # Stränga toleranser för hög precision
                max_step = h,
                dense_output=True)       # Möjliggör sampling av fler punkter

T = sol.t
Y = sol.y.T  # Transponera så det matchar din tidigare struktur

'''







r1_traj = Y[:, 0:3]     # x,y,z för kropp 1
r2_traj = Y[:, 6:9]     # kropp 2
r3_traj = Y[:, 12:15]   # kropp 3


# Skapa 3D figur 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sätt axelgränser (justera beroende på dina startvärden)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

# Initiera punkter och spår
point1, = ax.plot([], [], [], 'ro', label='Body 1')
point2, = ax.plot([], [], [], 'go', label='Body 2')
point3, = ax.plot([], [], [], 'bo', label='Body 3')

trail1, = ax.plot([], [], [], 'r-', alpha=0.5)
trail2, = ax.plot([], [], [], 'g-', alpha=0.5)
trail3, = ax.plot([], [], [], 'b-', alpha=0.5)

ax.legend()

# Lägg till slider för hastighet
axspeed = plt.axes([0.2, 0.01, 0.65, 0.03])
speed_slider = Slider(axspeed, 'Speed', 1, 100, valinit=10, valstep=1)
speed_factor = [10]

def update_speed(val):
    speed_factor[0] = int(speed_slider.val)
speed_slider.on_changed(update_speed)

# Initieringsfunktion
def init():
    point1.set_data([], [])
    point1.set_3d_properties([])
    point2.set_data([], [])
    point2.set_3d_properties([])
    point3.set_data([], [])
    point3.set_3d_properties([])
    return point1, point2, point3, trail1, trail2, trail3

# Uppdateringsfunktion
def update(frame):
    frame = frame * speed_factor[0]
    if frame >= len(T):
        frame = len(T)-1
    x1, y1, z1 = r1_traj[frame]
    x2, y2, z2 = r2_traj[frame]
    x3, y3, z3 = r3_traj[frame]

    point1.set_data([x1], [y1])
    point1.set_3d_properties([z1])
    point2.set_data([x2], [y2])
    point2.set_3d_properties([z2])
    point3.set_data([x3], [y3])
    point3.set_3d_properties([z3])

    trail1.set_data(r1_traj[:frame, 0], r1_traj[:frame, 1])
    trail1.set_3d_properties(r1_traj[:frame, 2])
    trail2.set_data(r2_traj[:frame, 0], r2_traj[:frame, 1])
    trail2.set_3d_properties(r2_traj[:frame, 2])
    trail3.set_data(r3_traj[:frame, 0], r3_traj[:frame, 1])
    trail3.set_3d_properties(r3_traj[:frame, 2])

    return point1, point2, point3, trail1, trail2, trail3

ani = FuncAnimation(fig, update, frames=len(T), init_func=init,
                    blit=True, interval=20, repeat=True)
plt.show()





# Metrics för mätningar

start_ref = time.time()

# Teoretiskt perfekt referenslösning med DOP853 metod
ref_sol = solve_ivp(fun=lambda t, y: deriv(t, y, m, G),
                    t_span=(t0, t_max),
                    y0=y0,
                    method='DOP853',
                    rtol=1e-12, atol=1e-14,
                    dense_output=True)

ref_Y = ref_sol.sol(T).T
end_ref = time.time()
ref_time = end_ref - start_ref

# Beräknar fel 
errors = np.linalg.norm(Y - ref_Y, axis=1)
max_error = np.max(errors)
mean_error = np.mean(errors)

# Chaotic divergence tidsgräns, storlek på fel 
threshold = 0.5
chaotic_idx = np.where(errors > threshold)[0]
if len(chaotic_idx) > 0:
    chaotic_time = T[chaotic_idx[0]]
else:
    chaotic_time = t_max


print("\n--- Simulation Results ---")
print(f"Method: CUSTOM STEP FUNCTION")
print(f"Reference runtime (DOP853): {ref_time:.2f} s")
print(f"Simulation points: {len(T)}")
print(f"Maximum Error (vs reference): {max_error:.2e}")
print(f"Mean Error (vs reference): {mean_error:.2e}")
print(f"Time Until Chaotic Divergence (> {threshold}): {chaotic_time:.2f}")
print("--------------------------\n")

