import numpy as np
from numpy.linalg import eig
from scipy.special import hankel2

def theodorsen(k):
    """Compute Theodorsen’s function C(k)."""
    if k == 0:
        return 1.0
    H1 = hankel2(1, k)
    H0 = hankel2(0, k)
    return H1 / (H1 + 1j * H0)

def H_of_k(k, rho, V, b, cma=0.25):
    """
    Aerodynamic influence matrix for a 2-DOF typical section (plunge h, pitch α).
    Thin-airfoil unsteady theory with Theodorsen’s function.
    """
    Ck = theodorsen(k)

    # Unsteady lift & moment coefficients (standard typical section form)
    # Generalized forces in plunge (h) and pitch (α)
    # This matrix form comes from Theodorsen theory for a 2D flat plate

    # Dynamic pressure
    q = 0.5 * rho * V**2

    # Aerodynamic influence matrix (2x2)
    # h equation (lift per plunge), α equation (moment per pitch)
    H = np.array([
        [2*np.pi*Ck,        2*np.pi*(0.5 - cma)*Ck],
        [2*np.pi*(0.5 - cma)*Ck, 2*np.pi*((0.5 - cma)**2 + 0.125)*Ck]
    ])

    return q * H

def pk_flutter(M, K, C, rho, b, V_range, tol=1e-6, max_iter=50):
    n = M.shape[0]
    results = []

    for V in V_range:
        # Guess initial frequency from structural modes
        eigvals, _ = eig(np.linalg.inv(M) @ K)
        omega_guess = np.sqrt(np.real(eigvals))
        k_guess = omega_guess * b / V

        for mode in range(len(omega_guess)):
            k = k_guess[mode]
            for it in range(max_iter):
                # Aerodynamic matrix
                Hk = H_of_k(k, rho, V, b)

                # Effective stiffness including aero
                A = -K + Hk
                B = C
                Cmat = -M

                # First-order form
                bigA = np.block([
                    [np.zeros((n,n)), np.eye(n)],
                    [np.linalg.inv(Cmat) @ A, np.linalg.inv(Cmat) @ B]
                ])

                eigvals_big, _ = eig(bigA)
                # Pick closest mode
                idx = np.argmin(np.abs(np.imag(eigvals_big) - omega_guess[mode]))
                p = eigvals_big[idx]

                # Update k
                k_new = np.imag(p) * b / V
                if abs(k_new - k) < tol:
                    break
                k = k_new

            # Damping ratio
            damping = np.real(p) / np.abs(p)
            results.append((V, np.imag(p), damping))

    return results

# Example: typical section parameters
c    = 0.3925 # m
b    = c/2

m    = 2.167/2 + 7.5 # kg
x_e  = 0.105 # m
x_ca = c/4 # m
inercia = 0.4322 / 2# kg m^2
kh = 10 / (1.345/1000) # N/m
ka = 454 # Nm/rad
Cla = 4.660


M = np.array([[m                    , -m * b * (x_e - x_ca)],
              [-m * b * (x_e - x_ca), inercia              ]])     # mass in plunge & inertia in pitch

K = np.array([[kh , 0.0],
              [0.0, ka]])    # stiffness in plunge & pitch
C = np.zeros((2,2))            # no structural damping

rho = 1.225   # air density (kg/m³)
b   = 0.5     # semichord (m)
V_range = np.linspace(5, 100, 40)

results = pk_flutter(M, K, C, rho, b, V_range)

# Find flutter speed (where damping crosses 0)
flutter_speed = None
for V, freq, damp in results:
    if damp >= 0:  # instability
        flutter_speed = V
        break

print("Estimated flutter speed:", flutter_speed, "m/s")
