import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-poster")


alpha = 1.0
beta  = 0.5
gamma = 0.1
eps   = 0.2
phi   = 0.3
zeta  = 0.05
eta   = 1.0

# # Shorthand for spatial derivative
# Du = u.dx(0)
# Dv = v.dx(0)
 
# # Coefficient groups
# A = alpha + beta * u + gamma * u**2      # multiplies  u''  →  -A * u'v'
# B = eps   + phi   * u + zeta  * u**2    # multiplies  u'   →  +B * u'v
# C = eta                                  # multiplies  u    →  +C * u v

z = np.linspace(-2, 1)
sol = solve_ivp(lambda t, s: np.cos(t), [z[0], z[-1]], [1], t_eval=z)

plt.plot(sol.t, sol.y.flatten())
plt.show()
