import math

# region functions
def simpson_integration(fcn, a, b, n):
    """
    This function performs numerical integration using the Simpson's rule.
    :param fcn: the callback function
    :param a: left limit of integration
    :param b: right limit of integration
    :param n: number of subintervals (must be an even number)
    :return: the approximate value of the integral
    """
    h = (b - a) / n
    result = fcn(a) + fcn(b)

    for i in range(1, n):
        x = a + i * h
        factor = 4 if i % 2 != 0 else 2
        result += factor * fcn(x)

    result *= h / 3.0
    return result

def gamma(m):
    """
    This function calculates the gamma function using numerical integration.
    :param m: input value
    :return: gamma function value at the given input
    """
    # Use simpson_integration for numerical integration
    integrand = lambda t: t**(m - 1) * math.exp(-t)
    result = simpson_integration(integrand, 0, 50, 10000)

    return result

def tPDF(m, u):
    """
    This function calculates the t-distribution cumulative distribution function (CDF).
    :param m: degrees of freedom
    :param u: value for which CDF is calculated
    :return: t-distribution CDF at the given value
    """
    # Calculate K_m using the provided formula
    gamma_1 = gamma(0.5 * m + 0.5)
    gamma_2 = gamma(0.5 * m)
    K_m = gamma_1 / (math.sqrt(m * math.pi) * gamma_2)

    # Calculate the t-distribution probability using the provided formula
    probability = K_m * simpson_integration(lambda x: (1 + (x**2) / m) ** (-0.5 * (m + 1)), -50, u, 10000)

    return probability

def FZ(args):
    """
    This function calculates F(z) based on the tPDF function.
    :param args: tuple containing degrees of freedom and z value
    :return: F(z) value
    """
    m, u = args
    return tPDF(m, u)

def main():
    get_out = False
    while not get_out:
        m = input("Degrees of Freedom (integer): ")
        u = input("Upper integration Limit (float): ")
        m = int(m)
        u = float(u)
        Fz = FZ((m, u))
        print("F({:.3f}) = {:.3f}".format(u, Fz))
        get_out = input("Go Again (Yes or No)?").lower() == "no"

if __name__ == '__main__':
    main()
