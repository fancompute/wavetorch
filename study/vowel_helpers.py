
from wavetorch import WaveSource, WaveIntensityProbe

def setup_src_coords(src_x, src_y, Nx, Ny, Npml):
    if (src_x is not None) and (src_y is not None):
        # Coordinate are specified
        return [WaveSource(src_x, src_y)]
    else:
        # Center at left
        return [WaveSource(Npml + 20, int(Ny / 2))]


def setup_probe_coords(N_classes, px, py, pd, Nx, Ny, Npml):
    if (py is not None) and (px is not None):
        # All probe coordinate are specified
        assert len(px) == len(py), "Length of px and py must match"

        return [WaveIntensityProbe(px[j], py[j]) for j in range(0, len(px))]

    if (py is None) and (pd is not None):
        # Center the probe array in y
        span = (N_classes - 1) * pd
        y0 = int((Ny - span) / 2)
        assert y0 > Npml, "Bottom element of array is inside the PML"
        y = [y0 + i * pd for i in range(N_classes)]

        if px is not None:
            assert len(px) == 1, "If py is not specified then px must be of length 1"
            x = [px[0] for i in range(N_classes)]
        else:
            x = [Nx - Npml - 20 for i in range(N_classes)]

        return [WaveIntensityProbe(x[j], y[j]) for j in range(0, len(x))]

    raise ValueError("px = {}, py = {}, pd = {} is an invalid probe configuration".format(pd))
