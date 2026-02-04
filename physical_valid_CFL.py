def bulk_modulus_check(dt, dx, dense):
    c = dx / dt
    K = c ** 2 * dense
    return K

def E_mu_check(dt, dx, dense, E, mu):
    K = E / (3 * (1 - 2 * mu))
    c = dx / dt
    _K = c ** 2 * dense
    return K, _K

# print(bulk_modulus_check(5e-5, 1 / 128, 1000))
print(E_mu_check(5e-5, 1 / 128, 1000, 5e4, 0.41))