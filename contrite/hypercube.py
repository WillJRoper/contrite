import h5py
import numpy as np
from unyt import yr, Gyr, Myr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy

class HyperCube():

    def __init__(self, props=None, prop_names=None, dims=None, lam=None,
                 nu=None, cube_path=None):

        # Are we making new grid or reading an existing one?
        if cube_path is not None:
            self.load_hypercube(cube_path)
        else:

            self.ndim = len(dims)

            self.props = props
            for name in prop_names:
                setattr(self, name, props[name])
                
            self.lam = lam
            self.nu = nu

        # Define an attribute to store the hdf5 object 
        self.hdf = None

        # List property names in hypercube dimension order
        self.prop_names = prop_names

        # Compute the width of bins along each dimension
        for name in prop_names:
            setattr(self, name + "_width",
                    getattr(self, name)[1] - getattr(self, name)[0])

    def get_hypercube_spectra(self, k, **kwargs):

        # Intialise an array to hold the spectra
        spectra = np.zeros(self.lam.size)

        # Loop until we've found all spectra from the hypercube
        for ind in range(k):

            # Set up grid slices
            grid_tup = [slice(None)] * self.ndim

            # Loop over properties and find the index in the hypercube for each.
            for prop_ind, name in enumerate(self.prop_names):

                # Get the property array
                prop_arr = getattr(self, name)
                prop_bin_width = getattr(self, name + "_width")

                # Find the index corresponding to this property
                if name == "max_age":
                    cube_ind = np.searchsorted(prop_arr, kwargs[name])
                else:
                    cube_ind =  np.searchsorted(prop_arr, kwargs[name][ind])

                # Handle the upper edge of the cube
                if cube_ind >= len(prop_arr):
                    cube_ind = len(prop_arr) - 1
                elif cube_ind < 0:
                    cube_ind = 0

                # Include this index 
                grid_tup[prop_ind] = cube_ind

            # Convert to a tuple. 
            grid_tup = tuple(grid_tup)

            # Include this spectra
            spectra += self.cube[grid_tup] * kwargs["mass"][ind]

        return spectra

    def load_hypercube(self, cube_path):

        # Open the file
        hdf = h5py.File(cube_path, "r")

        # Populate this cube from the file
        for key in hdf.keys():
            if key == "HyperCube":
                setattr(self, "cube", hdf[key][...])
            else:
                setattr(self, key, hdf[key][...])
        for key in hdf.attrs.keys():
            setattr(self, key, hdf.attrs[key])

        hdf.close()

    # def write_hypercube(self, cube_path):

    #     # Create the HDF5 file
    #     hdf = h5py.File(cube_path, "w")

    #     # Get the attributes of the cube
    #     attrs = [a for a in dir(obj) if not a.startswith('__')
    #              and not callable(getattr(obj, a))]

    #     # Store the cube as a dataset
    #     for a in attrs:
    #         arr = getattr(self, a)
    #         if isinstance(arr, np.ndarray):
    #             hdf.create_dataset(a, data=arr, dtype=arr.dtype,
    #                                shape=arr.shape, compression="gzip")
    #         else:
    #             hdf.attrs[a] = arr

    #     hdf.close


    def plot_dist(self):

        # Set up figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)

        for i, max_age in enumerate(self.max_age):
            print("Currently plotting for log_10(max_age)=%.2f" % max_age)
            for j, peak in enumerate(self.peak):

                # Skip peaks beyond the max age
                if peak >= max_age:
                    continue
            
                for k, tau in enumerate(self.tau):
                    for l, metal in enumerate(self.metallicity):

                        ax.loglog(self.lam, self.cube[i, j, k, l, :], alpha=0.1,
                                color="k")

        ax.set_xlabel('$\lambda / [\AA]$')
        ax.set_ylabel('$L_{\mathrm{nu}} M_\star^{-1}/$ [erg / s / Hz/ M$_\odot$]')
    
        fig.savefig("hypercube_spectra.png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        

def create_hypercube(grid, props, prop_names, dims, cube_path, rank, nranks,
                     comm):

    # Initialise the hypercube
    hcube = HyperCube(props=props, prop_names=prop_names, dims=dims,
                      lam=grid.lam, nu=grid.nu)

    # Extract properties
    # TODO: needs generalising
    max_ages = props["max_age"]
    peaks = props["peak"]
    taus = props["tau"]
    metals = props["metallicity"]

    # Create the list of parameters that describe the hypercube ready for
    # parallelising
    params = []
    for i, max_age in enumerate(max_ages):
        for j, peak in enumerate(peaks):

            # Skip peaks beyond the max age
            if peak >= max_age:
                continue
            
            for k, tau in enumerate(taus):
                for l, metal in enumerate(metals):

                    params.append([(i, j, k, l), (max_age, peak, tau, metal)])

    # Get the range for each rank
    if rank == 0:
        print("Distributing %d spectra over %d ranks" % (len(params), nranks))
    rank_bins = np.linspace(0, len(params), nranks + 1, dtype=int)

    # Get this ranks indices
    my_inds = np.arange(rank_bins[rank], rank_bins[rank + 1], 1, dtype=int)

    # Loop over parameters making SEDs
    spectra = []
    for ind in my_inds:

        (i, j, k, l), (max_age, peak, tau, metal) = params[ind]

        sfh_props = {'peak_age': 10 ** peak * yr, "tau": tau,
                     "max_age": 10 ** max_age * yr}
        Z_p = {'log10Z': metal}

        # Should we report what we have done?
        if (ind - rank_bins[rank]) % 1000 == 0:
            print("Rank %d done: %d of %d (%.2f)"
                  % (rank, ind - rank_bins[rank],
                     rank_bins[rank + 1] - rank_bins[rank],
                     (ind - rank_bins[rank]) /
                     (rank_bins[rank + 1] - rank_bins[rank]) * 100) + "%")

        # Define the functional form of the star formation and
        # metal enrichment histories
        sfh = SFH.LogNormal(sfh_props)
        
        # Constant metallicity
        Zh = ZH.deltaConstant(Z_p)
                    
        # Get the 2D star formation and metal enrichment history
        # for the given SPS grid. This is (age, Z).
        sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                             sfh, Zh, stellar_mass=1.0)
        
        # Make a galaxy from this SFZH
        galaxy = Galaxy(sfzh)
                    
        # Compute SED
        sed = galaxy.get_stellar_spectra(grid)

        # Store the results
        spectra.append([(i, j, k, l), (max_age, peak, tau, metal),
                        np.float32(sed._lnu)])

    print("Rank %d finished %d spectra"
          % (rank, rank_bins[rank + 1] - rank_bins[rank]))

    # Create the HDF5 file
    hdf = h5py.File(cube_path.replace("<rank>", str(rank)), "w")

    # Write out this ranks results
    for spec_lst in spectra:

        # Extract the contents
        (i, j, k, l), (max_age, peak, tau, metal), spec = spec_lst

        # Make a group for this spectra
        hdf.create_dataset("%d_%d_%d_%d" % (i, j, k, l),
                           data=spec, dtype=spec.dtype,
                           shape=spec.shape, compression="gzip")

    hdf.close()

    comm.barrier()

    if rank == 0:

        # Create a single file with a virtual dataset
        # NOTE: only works for h5py>=2.9
        hdf = h5py.File(cube_path.replace("_<rank>", ""), "w")

        # Get the attributes of the cube
        attrs = [a for a in dir(hcube) if not a.startswith('__')
                 and not callable(getattr(hcube, a))]

        # Store the cube as a dataset
        for a in attrs:
            if a == "hdf" or "prop" in a:
                continue
            arr = getattr(hcube, a)
            print("Writing:", a)
            if isinstance(arr, np.ndarray):
                hdf.create_dataset(a, data=arr, dtype=arr.dtype,
                                   shape=arr.shape, compression="gzip")
            else:
                hdf.attrs[a] = arr

        # Create the virtual dataset for the cube
        shape = [dims[prop] for prop in prop_names]
        shape.append(dims["wavelength"])
        shape = tuple(shape)
        virtual_cube = h5py.VirtualLayout(
            shape=shape, dtype=np.float32
        )

        # Loop over rank files and include their spectra in the virtual cube
        for r in range(nranks):

            # Open the HDF5 file
            rank_hdf = h5py.File(cube_path.replace("<rank>", str(r)), "r")

            # Loop over datasets converting them to virtual sources
            for key in rank_hdf.keys():
                i, j, k, l = key.split("_")
                i = int(i)
                j = int(j)
                k = int(k)
                l = int(l)
                virtual_spectra = h5py.VirtualSource(rank_hdf[key])
                virtual_cube[i, j, k, l, :] = virtual_spectra

            rank_hdf.close()

        # Store the hypercube in the master file
        hdf.create_virtual_dataset('HyperCube', virtual_cube, fillvalue=-5)

        hdf.close()
        
    return hcube
