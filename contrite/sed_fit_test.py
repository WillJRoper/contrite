import os
import time
import random
import numpy as np
import h5py
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import yr, Gyr, Myr, kpc, arcsec

import zeus

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.filters import FilterCollection as Filters
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from astropy.cosmology import Planck18 as cosmo
from synthesizer.dust import power_law
from synthesizer.sed import Sed
from multiprocessing import Pool

from mpi4py import MPI
import mpi4py
import numpy as np
from mpi4py import MPI

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
nranks = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


# Set the random seed
np.random.seed(42)


def logprior(theta, k, grid):
    ''' The natural logarithm of the prior probability. '''

    # Extract fitting parameters
    peaks = theta[: k]
    taus = theta[k: 2 * k]
    metallicity = theta[2 * k: 3 * k]
    masses = theta[3 * k: 4 * k]
    # max_age = theta[-1]

    lp = 0.

    # Uniform priors
    age_min, age_max = grid.log10ages.min(), grid.log10ages.max()
    tau_min, tau_max = 10**-5, 1
    Zmin, Zmax = grid.log10metallicities.min(), grid.log10metallicities.max()
    mass_min, mass_max = 4, 12

    # Set prior to 1 (log prior to 0) if in the range and zero (-inf)
    # outside the range
    # lp += 0. if age_min < max_age < age_max else -np.inf
    lp += 0. if np.all(np.logical_and(peaks <= age_max, peaks >= age_min)) else -np.inf
    lp += 0. if np.all(np.logical_and(taus <= tau_max, taus >= tau_min)) else -np.inf
    lp += 0. if np.all(np.logical_and(metallicity <= Zmax, metallicity >= Zmin)) else -np.inf
    lp += 0. if np.all(np.logical_and(masses <= mass_max, masses >= mass_min)) else -np.inf

    # Extra constraints

    # # The peak must be less than the maximum age
    # lp += 0. if np.all(peaks < max_age) else -np.inf

    return lp


def loglike(theta, data, sigma, cosmo, filters, k, grid):
    '''The natural logarithm of the likelihood.'''

    # Extract fitting parameters
    peaks = 10 ** theta[: k] * yr
    taus = theta[k: 2 * k]
    metallicity = theta[2 * k: 3 * k]
    masses = theta[3 * k: 4 * k]
    # max_age = 10 ** theta[-1] * yr

    sed, _ = get_spectra(peaks, taus, metallicity, masses, k)

    # Calculate photometry of the model
    photm = np.array([f.apply_filter(sed.fnu, sed.nuz) for f in filters])

    # return the log likelihood
    return -0.5 * np.sum(((photm - data) / sigma) ** 2)


def logpost(theta, k, data, sigma, cosmo, filters, grid):
    '''The natural logarithm of the posterior.'''

    prior = logprior(theta, k, grid)

    if prior > -np.inf:
        lnl = loglike(theta, data, sigma, cosmo,
                      filters, k, grid)
        return prior + lnl
    else:
        return -np.inf


def get_fake_photometry(grid, filters, n):

    # Generate random variables for the star formation events
    max_age = 100 * Myr
    peaks = np.random.uniform(1, max_age.value, n) * Myr
    taus = np.random.uniform(0, 1, n)
    metallicity = np.random.uniform(-2, -1, n)
    masses = np.random.uniform(5, 9, n)

    # Redshift
    z = 8

    # Start with a constant
    Z_p = {"Z": 0.01}
    Zh = ZH.deltaConstant(Z_p)
    sfh_p = {"duration": max_age}
    sfh = SFH.Constant(sfh_p)  # constant star formation
    tot_sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh,
                             stellar_mass=10**9)
    
    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.semilogy()

    ax.plot(grid.log10ages, tot_sfzh.sfh, label="Constant")

    # Define star formation history variables
    for ind in range(n):
        
        sfh_props = {'peak_age': peaks[ind], "tau": taus[ind],
                     "max_age": max_age}
        Z_p = {'log10Z': metallicity[ind]}
        stellar_mass = 10 ** masses[ind]

        # Define the functional form of the star formation and metal enrichment
        # histories
        sfh = SFH.LogNormal(sfh_props)
        
        Zh = ZH.deltaConstant(Z_p)  # constant metallicity

        # Get the 2D star formation and metal enrichment history for the given
        # SPS grid. This is (age, Z).
        sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                             sfh, Zh, stellar_mass=stellar_mass)
        
        ax.plot(grid.log10ages, sfzh.sfh,
                label=", ".join([key + "=%.2f" % val
                                 for key, val in sfh_props.items()]))

        tot_sfzh = tot_sfzh + sfzh

    ax.set_xlabel('$log_{10}$(age/[yr])')
    ax.set_ylabel('SFR / [M$_\odot$/yr]')
    ax.set_ylim(10**2, 10**8.5)

    total_stellar_mass = np.sum(tot_sfzh.sfh * grid.log10ages)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=1)

    fig.savefig("../SFHZ_split_test.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    fig1, ax1 = tot_sfzh.plot(show=False)
    
    fig1.savefig("../SFHZ_test.png")
    plt.close(fig1)

    # Make a galaxy from this SFZH
    galaxy = Galaxy(tot_sfzh)

    # Compute SED
    sed = galaxy.get_stellar_spectra(grid)
    sed.get_fnu0()
    print(tot_sfzh)
    print("log_10(Mass_*/M_sun)=", np.log10(total_stellar_mass))
    print(np.sum(sed.fnu))

    # Calculate photometry of the model
    photm = np.array([f.apply_filter(sed.fnu, sed.nuz) for f in filters])
    flams = np.array([f.pivwv() for f in filters])
    
    # Set up plot
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.loglog(sed.lamz, sed.fnu)
    ax.set_ylim(10**26., None)
    ax.set_xlim(10**2, 10**6)

    for i, f in enumerate(filters):
        ax.scatter(flams[i], photm[i], marker="+",
                   label=f.filter_code)

    ax.set_xlabel('$\lambda / [\AA]$')
    ax.set_ylabel('$F_{\mathrm{nu}}/$ [nJy]')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=1)
    
    fig.savefig("../compositeSFHZ_sed_test.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    return photm, sed

def get_spectra(peaks, taus, metallicity, masses, k):

    tot_sfzh = None

    # Loop over each star formation event
    for ind in range(k):
        
        sfh_props = {'peak_age': peaks[ind], "tau": taus[ind],
                     "max_age": peaks[ind] + 10 * Myr}
        Z_p = {'log10Z': metallicity[ind]}
        stellar_mass = 10 ** masses[ind]
        
        # Define the functional form of the star formation and metal enrichment
        # histories
        sfh = SFH.LogNormal(sfh_props)
        
        Zh = ZH.deltaConstant(Z_p)  # constant metallicity

        # Get the 2D star formation and metal enrichment history for the given
        # SPS grid. This is (age, Z).
        sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                             sfh, Zh, stellar_mass=stellar_mass)
        
        # Include this star formation event
        if tot_sfzh is None:
            tot_sfzh = sfzh
        else:
            tot_sfzh = tot_sfzh + sfzh

    # Make a galaxy from this SFZH
    galaxy = Galaxy(tot_sfzh)

    # Compute SED
    sed = galaxy.get_stellar_spectra(grid)
    sed.get_fnu0()

    return sed, tot_sfzh


class HyperCube():

    def __init__(self, props=None, cube=None, lam=None, cube_path=None):

        # Are we making new grid or reading an existing one?
        if cube_path is not None:
            self.load_hypercube(cube_path)
        else:

            self.ndim = len(props) + 1

            self.props = props
            for name, prop in props.items():
                setattr(self, name, prop)

            self.cube = cube

            self.lam = lam

    def get_value(self):

        grid_tup = [slice(None)] * self.ndim

        grid_tup = tuple(grid_tup)

        return cube[gird_tup]

    def load_hypercube(self, cube_path):

        # Open the file
        hdf = h5py.File(cube_path, "r")

        # Populate this cube from the file
        for key in hdf.keys():
            setattr(self, key, hdf[key][...])
        for key in hdf.attrs.keys():
            setattr(self, key, hdf.attrs[key])

        hdf.close()

    def write_hypercube(self, cube_path):

        # Create the HDF5 file
        hdf = h5py.File(cube_path, "w")

        # Get the attributes of the cube
        attrs = [a for a in dir(obj) if not a.startswith('__')
                 and not callable(getattr(obj, a))]

        # Store the cube as a dataset
        for a in attrs:
            arr = getattr(self, a)
            if isinstance(arr, np.ndarray):
                hdf.create_dataset(a, data=arr, dtype=arr.dtype,
                                   shape=arr.shape, compression="gzip")
            else:
                hdf.attrs[a] = arr

        hdf.close


    def plot_dist(self):

        # Set up figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, max_age in enumerate(self.max_age):
            for j, peak in enumerate(self.peak):

                # Skip peaks beyond the max age
                if peak > max_age:
                    continue
            
                for k, tau in enumerate(self.tau):
                    for l, metal in enumerate(self.metallicity):

                        ax.plot(self.lam, self.cube[i, j, k, l, :], alpha=0.1,
                                color="k")

        ax.set_xlabel('$\lambda / [\AA]$')
        ax.set_ylabel('$F_{\mathrm{nu}}/$ [nJy]')
    
        fig.savefig("../hypercube_spectra.png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        

def create_hypercube(grid, out_path):

    # Get useful constants
    nlam = grid.lam.size
    nZ = grid.log10metallicities.size
    nage = grid.log10ages.size

    # Set up the hypercube array
    # Note: this structure is (max_age, peaks, taus, Zs, wavelength)
    if rank == 0:
        cube = np.zeros((nage, nage, nage, nZ, nlam))

    # Define each dimension of the cube
    max_ages = grid.log10ages
    peaks = grid.log10ages
    taus = np.linspace(0, 1, nage)
    metals = grid.log10metallicities

    # Set up hypercube object
    if rank == 0:
        props  = {"max_age": max_ages, "peak": peaks, "tau": taus,
                  "metallicity": metals}
        hcube = HyperCube(props=props, cube=cube, lam=grid.lam)

    # Create the list of parameters that describe the hypercube ready for
    # parallelising
    params = []
    for i, max_age in enumerate(max_ages):
        for j, peak in enumerate(peaks):

            # Skip peaks beyond the max age
            if peak > max_age:
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

        sfh_props = {'peak_age': 10 ** peak * Myr, "tau": tau,
                     "max_age": 10 ** max_age * Myr}
        Z_p = {'log10Z': metal}

        # Should we report what we have done?
        if j == 0 and k == 0 and l == 0:
            print("Rank %d is computing spectra for log10(max_age)[%d]=%.2f"
                  % (rank, i, max_age))

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
        sed.get_fnu0()

        # Store the results
        spectra.append([(i, j, k, l), sed._fnu])

    print("Rank %d finished %d spectra"
          % (rank, rank_bins[rank + 1] - rank_bins[rank]))

    # Collect the spectra
    collection = comm.gather(spectra, root=0)

    # Store spectra in the hypercube
    if rank == 0:

        for lst in collection:
            for spec_tup in lst:

                # Extract the results for this spectra
                (i, j, k, l) = spec_tup[0]
                spec = spec_tup[1]

                # Store this spectra
                hcube.cube[i, j, k, l, :] = sed.fnu

        # Write out the hypercube
        hcube.write_hypercube(out_path)

    # # Loop over cube dimensions populating SEDs
    # for i, max_age in enumerate(max_ages):
    #     print("Computing spectra for log10(max_age)[%d]=%.2f" % (i, max_age))
    #     for j, peak in enumerate(peaks):

    #         # Skip peaks beyond the max age
    #         if peak > max_age:
    #             continue
            
    #         for k, tau in enumerate(taus):
    #             for l, metal in enumerate(metals):
                    
    #                 sfh_props = {'peak_age': 10 ** peak * Myr, "tau": tau,
    #                              "max_age": 10 ** max_age * Myr}
    #                 Z_p = {'log10Z': metal}

    #                 # Define the functional form of the star formation and
    #                 # metal enrichment histories
    #                 sfh = SFH.LogNormal(sfh_props)

    #                 # Constant metallicity
    #                 Zh = ZH.deltaConstant(Z_p)
                    
    #                 # Get the 2D star formation and metal enrichment history
    #                 # for the given SPS grid. This is (age, Z).
    #                 sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
    #                                      sfh, Zh, stellar_mass=1.0)

    #                 # Make a galaxy from this SFZH
    #                 galaxy = Galaxy(sfzh)

    #                 # Compute SED
    #                 sed = galaxy.get_stellar_spectra(grid)
    #                 sed.get_fnu0()

    #                 # Store this spectra
    #                 hcube.cube[i, j, k, l, :] = sed.fnu
    
    return hcube


if __name__ == "__main__":

    # Define the grid
    grid_name = "bc03_chabrier03-0.1,100"
    # grid_dir = "/Users/willroper/Documents/University/Synthesizer" \
    #     "/synthesizer_data/grids/"
    grid_dir = "/cosma/home/dp004/dc-rope1/cosma8/CONTRITE/data/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    filter_codes = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W",
                    "JWST/NIRCam.F200W", "JWST/NIRCam.F277W",
                    "JWST/NIRCam.F444W"]
    filters = Filters(filter_codes=filter_codes, new_lam=grid.lam)
    if rank == 0:
        photm, true_sed = get_fake_photometry(grid, filters, n=10)

    # Intialise the Hypercube
    cube_path = "../hypercube.hdf5"

    t = time.time()
    hcube = create_hypercube(grid, cube_path)
    print("Hypercube creation took:", time.time() - t)

    if rank == 0:
        t = time.time()
        hcube = HyperCube(cube_path)
        print("Hypercube reading took:", time.time() - t)

        hcube.plot_dist()

        k = 2
        
        # Define the number of elements in each dimension
        dims = np.array([k, k, k, k])

        ndim = np.sum(dims)  # Number of parameters/dimensions
        # Number of walkers to use. It should be at least twice the number of dimensions.
        nwalkers = 20
        nsteps = 500  # Number of steps/iterations.
    
        # Initial positions of the walkers.
        print(grid.log10ages.min(), grid.log10ages.max())
        print(grid.log10metallicities.min(), grid.log10metallicities.max())
        start = np.zeros((nwalkers, ndim))
        start[:, : k] = np.log10((50 * Myr).to(yr).value)
        start[:, k: 2 * k] = 0.5
        start[:, 2 * k: 3 * k] = -2
        start[:, 3 * k: 4 * k] = 9
        print(np.min(start, axis=0))
        print(np.max(start, axis=0))
        start[:, : k] += start[:, : k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        start[:, k: 2 * k] += start[:, k: 2 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        start[:, 2 * k: 3 * k] += start[:, 2 * k: 3 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        start[:, 3 * k: 4 * k] += start[:, 3 * k: 4 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        
        print(np.min(start, axis=0))
        print(np.max(start, axis=0))
        
        t = time.time()
        prob = loglike(start[0, :], photm, 0.1, cosmo, filters, k, grid)
        print("Test gave P=%.2e and took %.4f s" % (prob, time.time() - t))

    # Read the cube back in
    
    # sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost,
    #                                args=[k, photm, 10**27, cosmo, filters, grid],
    #                                maxiter=10**4)

    # # Run MCMC
    # sampler.run_mcmc(start, nsteps)

    # # Initialise the sampler
    # sampler.summary  # Print summary diagnostics

    # # Make labels
    # labels = np.zeros(ndim, dtype=object)
    # labels[:k] = ["peak_age_%d" % i for i in range(k)]
    # labels[k: 2 * k] = ["$\tau_%d$" % i for i in range(k)]
    # labels[2 * k: 3 * k] = ["$Z_%d$" % i for i in range(k)]
    # labels[3 * k: 4 * k] = ["$M_{\star,%d}$" % i for i in range(k)]

    # # Plot the walkers
    # plt.figure(figsize=(16, 1.5 * ndim))
    # for n in range(ndim):
    #     plt.subplot2grid((ndim, 1), (n, 0))
    #     plt.plot(sampler.get_chain()[:,:,0], alpha=0.5)
    #     plt.ylabel(labels[n])
    # plt.tight_layout()
    # plt.savefig("../walkers.png", bbox_inches="tight", dpi=100)
    # plt.close()

    # # Flatten the chains, thin them by a factor of 10, and remove the burn-in
    # # (first half of the chain)
    # chain = sampler.get_chain(flat=True, discard=50, thin=10)

    # # Plot marginal posterior distributions
    # fig, axes = zeus.cornerplot(chain, labels=labels)
    # fig.savefig("../corner_plot.png", bbox_inches="tight", dpi=100)
    # plt.close(fig)

    # # Get median parameters
    # chain = sampler.get_chain(flat=False, discard=50, thin=10)
    # fit_params = np.median(chain, axis=0)

    # # Get the fit
    # fit_sed, fit_sfzh = get_spectra(fit_params[:k] * yr, fit_params[k: 2 * k],
    #                                 fit_params[2 * k: 3 * k],
    #                                 fit_params[3 * k: 4 * k], k)
    
    # # Set up plot
    # fig = plt.figure(figsize=(6, 3))
    # ax = fig.add_subplot(111)
    # ax.grid(True)

    # for i in range(chain.shape[0]):
    #     sed, _ = get_spectra(chain[i, :k] * yr,
    #                          chain[i, k: 2 * k],
    #                          chain[i, 2 * k: 3 * k],
    #                          chain[i, 3 * k: 4 * k], k)
    #     ax.loglog(true_sed.lamz, sed.fnu, color="k", alpha=0.3)
    
    # ax.loglog(true_sed.lamz, true_sed.fnu, color="g",
    #           linestyle="--", label="Truth")
    # ax.loglog(true_sed.lamz, fit_sed.fnu, color="c", label="Fit")
    # ax.set_ylim(10**26., None)
    # ax.set_xlim(10**2, 10**6)

    # ax.set_xlabel('$\lambda / [\AA]$')
    # ax.set_ylabel('$F_{\mathrm{nu}}/$ [nJy]')

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    #           fancybox=True, shadow=True, ncol=2)
    
    # fig.savefig("../fit_sed_test.png", bbox_inches="tight", dpi=100)
    # plt.close(fig)

    # fig1, ax1 = fit_sfzh.plot(show=False)
    
    # fig1.savefig("../fit_SFHZ_test.png")
    # plt.close(fig1)

    
