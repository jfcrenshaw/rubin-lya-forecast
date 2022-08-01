"""Build a CCL tracer."""
import numpy as np
import pyccl as ccl


def build_tracer(b: float, rsd: float, nz: tuple, cosmo: ccl.Cosmology) -> ccl.Tracer:
    """Build a CCL tracer.

    Parameters
    ----------
    b: float
        The linear bias.
    rsd: float
        The RSD scale. Equals b * beta.
    nz: tuple
        The redshift kernel W(z). nz[0] is the redshift grid and nz[1] is W(z)
        defined on that grid.
    cosmo: ccl.Cosmology
        The CCL object defining the cosmology.

    Returns
    -------
    ccl.Tracer
        The CCL tracer.
    """
    # first create an empty tracer
    tracer = ccl.Tracer()

    # generate the radial kernel
    kernel = ccl.get_density_kernel(cosmo, nz)

    # we need a transfer function for the bias.
    # Since Anze said to set the bias to a constant,
    # I think I just need to set this constant too?
    a = (1.0 / (1 + nz[0]))[::-1]  # scale factor must be increasing
    transfer_a = (a, b * np.ones_like(a))

    # now we can add the density contribution to the tracer:
    tracer.add_tracer(cosmo, kernel=kernel, transfer_a=transfer_a)

    # Now we can add RSD
    # RSD uses the second derivative of the Bessel function
    # Anze also said set this constant?
    transfer_a_rsd = (a, rsd * np.ones_like(a))
    tracer.add_tracer(cosmo, kernel=kernel, transfer_a=transfer_a_rsd, der_bessel=2)

    return tracer
