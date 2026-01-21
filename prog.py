import numpy as np
import matplotlib.pyplot as plt


g = 9.81  

def model_step_euler(x, y, vx, vy, k, dt):

    ax = -k * vx
    ay = -k * vy - g
    

    x_new = x + vx * dt
    y_new = y + vy * dt
    
 
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    
    return x_new, y_new, vx_new, vy_new

def model_step_rk2(x, y, vx, vy, k, dt):
   
    ax1 = -k * vx
    ay1 = -k * vy - g
    
    vx_half = vx + ax1 * (dt / 2)
    vy_half = vy + ay1 * (dt / 2)

    ax2 = -k * vx_half
    ay2 = -k * vy_half - g
    
    
    x_new = x + vx_half * dt  
    y_new = y + vy_half * dt
    
    vx_new = vx + ax2 * dt
    vy_new = vy + ay2 * dt
    
    return x_new, y_new, vx_new, vy_new

def simulate_trajectory(v0, angle_deg, h0, k, dt=0.01, method='euler'):
   
    angle_rad = np.radians(angle_deg)
    x = 0
    y = h0
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    x_hist = [x]
    y_hist = [y]
    
    while y >= 0:
        if method == 'euler':
            x, y, vx, vy = model_step_euler(x, y, vx, vy, k, dt)
        elif method == 'rk2':
            x, y, vx, vy = model_step_rk2(x, y, vx, vy, k, dt)
        
        x_hist.append(x)
        y_hist.append(y)
        
        if len(x_hist) > 10000:
            break
            
    return x_hist, y_hist


v0 = 25.0       
angle = 45.0    
h0 = 0.0       
k_values = [0.0, 0.1, 0.3, 0.8] 

plt.figure(figsize=(10, 6))

for k in k_values:
    xs, ys = simulate_trajectory(v0, angle, h0, k, dt=0.05, method='euler')
    label = f'k={k} (Эйлер)' if k != 0 else 'k=0 (Вакуум)'
    plt.plot(xs, ys, label=label)

plt.title(f'Траектория полета (v0={v0} м/с, a={angle}°)\nЗависимость от сопротивления среды k')
plt.xlabel('Расстояние x (м)')
plt.ylabel('Высота y (м)')
plt.axhline(0, color='black', linewidth=1) # земля
plt.grid(True)
plt.legend()
plt.show()


k_test = 0.5
dt_test = 0.1

xs_euler, ys_euler = simulate_trajectory(v0, angle, h0, k_test, dt=dt_test, method='euler')
xs_rk2, ys_rk2 = simulate_trajectory(v0, angle, h0, k_test, dt=dt_test, method='rk2')

xs_ref, ys_ref = simulate_trajectory(v0, angle, h0, k_test, dt=0.001, method='rk2')

plt.figure(figsize=(10, 6))
plt.plot(xs_euler, ys_euler, 'r--', label='Метод Эйлера (dt=0.1)')
plt.plot(xs_rk2, ys_rk2, 'b--', label='Метод Р-К 2 (dt=0.1)')
plt.plot(xs_ref, ys_ref, 'g-', alpha=0.5, linewidth=2, label='Точное решение (dt=0.001)')

plt.title(f'Сравнение численных методов (k={k_test})')
plt.xlabel('Расстояние x (м)')
plt.ylabel('Высота y (м)')
plt.axhline(0, color='black', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()