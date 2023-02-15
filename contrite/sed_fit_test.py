import os
import time
import random
import numpy as np
import h5py
import sys
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

from hypercube import create_hypercube, HyperCube

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
    max_age = theta[-1]

    lp = 0.

    # Uniform priors
    age_min, age_max = grid.log10ages.min(), grid.log10ages.max()
    tau_min, tau_max = 10**-5, 5
    Zmin, Zmax = grid.log10metallicities.min(), grid.log10metallicities.max()
    mass_min, mass_max = 4, 12

    # Set prior to 1 (log prior to 0) if in the range and zero (-inf)
    # outside the range
    lp += 0. if age_min < max_age < age_max else -np.inf
    lp += 0. if np.all(np.logical_and(peaks <= age_max, peaks >= age_min)) else -np.inf
    lp += 0. if np.all(np.logical_and(taus <= tau_max, taus >= tau_min)) else -np.inf
    lp += 0. if np.all(np.logical_and(metallicity <= Zmax, metallicity >= Zmin)) else -np.inf
    lp += 0. if np.all(np.logical_and(masses <= mass_max, masses >= mass_min)) else -np.inf

    # Extra constraints

    # The peak must be less than the maximum age
    lp += 0. if np.all(peaks < max_age) else -np.inf

    # # Penalise the parmaeter set for large differences between max age and
    # # peak age
    # lp += np.sum(0.5 * ((peaks - max_age) / 10 ** 6) ** 2)

    return lp
    

def loglike(theta, data, sigma, cosmo, filters, k, grid, hcube):
    '''The natural logarithm of the likelihood.'''

    # Extract fitting parameters
    peaks = theta[: k]
    taus = theta[k: 2 * k]
    metallicity = theta[2 * k: 3 * k]
    masses = 10 ** theta[3 * k: 4 * k]
    z = theta[-1]

    sed = hcube.get_hypercube_spectra(k, max_age=max_age, peak=peaks, tau=taus,
                                      metallicity=metallicity, mass=masses)

    # Compute the luminosity distance
    luminosity_distance = cosmo.luminosity_distance(
            z).to('cm').value  # the luminosity distance in cm

    nuz = hcube.nu

    # Calculate photometry of the model
    if filters is not None:
        fit = np.array([f.apply_filter(sed, nuz) for f in filters])
    else:
        fit = sed
        
    # return the log likelihood
    return -0.5 * np.sum(((fit - data) / sigma) ** 2)


def logpost(theta, k, data, sigma, cosmo, filters, grid, hcube):
    '''The natural logarithm of the posterior.'''

    prior = logprior(theta, k, grid)

    if np.isfinite(prior):
        lnl = loglike(theta, data, sigma, cosmo,
                      filters, k, grid, hcube)
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

    total_stellar_mass = np.trapz(tot_sfzh.sfh, x=10 ** grid.log10ages)

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
    print("log_10(Mass_*/M_sun)=", np.log10(total_stellar_mass),
          np.log10(10 ** 9 + sum([10 ** m for m in masses])))
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

def get_spectra(peaks, taus, metallicity, masses, max_age, k):

    tot_sfzh = None

    # Loop over each star formation event
    for ind in range(k):
        
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


if __name__ == "__main__":

    # Define the grid
    grid_name = "bc03_chabrier03-0.1,100"
    grid_dir = "/Users/willroper/Documents/University/Synthesizer" \
        "/synthesizer_data/grids/"
    # grid_dir = "/cosma/home/dp004/dc-rope1/cosma8/CONTRITE/data/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    if rank == 0:
        filter_codes = ["HST/WFC3_UVIS1.F218W", "HST/WFC3_UVIS1.F390W",
                        "HST/WFC3_UVIS2.F438W", "HST/WFC3_UVIS2.F555W",
                        "JWST/NIRCam.F090W", "JWST/NIRCam.F150W",
                        "JWST/NIRCam.F200W", "JWST/NIRCam.F277W",
                        "JWST/NIRCam.F444W"]
        filters = Filters(filter_codes=filter_codes, new_lam=grid.lam)
        photm, true_sed = get_fake_photometry(grid, filters, n=2)

    # Intialise the Hypercube
    cube_path = sys.argv[1] + "/hypercube_<rank>.hdf5"

    # Define the hypercube dimensions
    dims = {"max_age": 50, "peak": 50, "tau": 50, "metallicity": 6,
            "wavelength": grid.lam.size}

    # Define each dimension of the cube
    max_ages = np.linspace(grid.log10ages[0], grid.log10ages[-1],
                           dims["max_age"])
    peaks = np.linspace(grid.log10ages[0], grid.log10ages[-1], dims["peak"])
    taus = np.linspace(10**-5, 5, dims["tau"])
    metals = np.linspace(grid.log10metallicities[0],
                         grid.log10metallicities[-1], dims["metallicity"])

    # Set up hypercube object
    props  = {"max_age": max_ages, "peak": peaks, "tau": taus,
              "metallicity": metals}
    prop_names = tuple(props.keys())

    # t = time.time()
    # hcube = create_hypercube(grid, props, prop_names, dims, cube_path,
    #                          rank, nranks, comm)
    # print("Hypercube creation took:", time.time() - t)

    if rank == 0:

        # Intialise the Hypercube
        cube_path = sys.argv[1] + "/hypercube.hdf5"
        
        t = time.time()
        hcube = HyperCube(prop_names=prop_names, cube_path=cube_path)
        print("Hypercube reading took:", time.time() - t)

        # hcube.plot_dist()

        k = 1
        
        # Define the number of elements in each dimension
        dims = np.array([k, k, k, k, 1])

        ndim = np.sum(dims)  # Number of parameters/dimensions
        # Number of walkers to use. It should be at least twice the number of dimensions.
        nwalkers = 100
        nsteps = 2000  # Number of steps/iterations.
    
        # Initial positions of the walkers.
        print(grid.log10ages.min(), grid.log10ages.max())
        print(grid.log10metallicities.min(), grid.log10metallicities.max())
        start = np.zeros((nwalkers, ndim))
        start[:, k: 2 * k] = 0.5
        start[:, 2 * k: 3 * k] = -2
        start[:, 3 * k: 4 * k] = 9
        start[:, -1] = 8
        print(np.min(start, axis=0))
        print(np.max(start, axis=0))
        start[:, -1] += start[:, -1] * np.random.uniform(-0.2, 0.2, nwalkers)
        for i in range(k):
            start[:, i] = start[:, -1]
        start[:, : k] -= start[:, -1][:, None] * np.random.uniform(0, 0.2, (nwalkers, k))
        start[:, k: 2 * k] += start[:, k: 2 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        start[:, 2 * k: 3 * k] += start[:, 2 * k: 3 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        start[:, 3 * k: 4 * k] += start[:, 3 * k: 4 * k] * np.random.uniform(-0.2, 0.2, (nwalkers, k))
        
        print(np.min(start, axis=0))
        print(np.max(start, axis=0))
        
        t = time.time()
        prob = loglike(start[0, :], photm, photm * 0.1, cosmo, filters, k, grid, hcube)
        print("Test gave P=%.2e and took %.4f s" % (prob, time.time() - t))
    
        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, logpost,
            args=[k, true_sed._fnu, true_sed._fnu * 0.1, cosmo, None, grid, hcube],
            maxiter=10**4)

        # Run MCMC
        sampler.run_mcmc(start, nsteps)

        # Get the burnin samples
        burnin = sampler.get_chain()

        # Set the new starting positions of walkers based on their last positions
        start = burnin[-1]

        sampler = zeus.EnsembleSampler(
            nwalkers, ndim, logpost,
            args=[k, true_sed._fnu, true_sed._fnu * 0.1, cosmo, None, grid, hcube],
            maxiter=10**4, moves=zeus.moves.GlobalMove())

        # Run MCMC
        sampler.run_mcmc(start, nsteps)

        # Initialise the sampler
        sampler.summary  # Print summary diagnostics

        # Make labels
        labels = np.zeros(ndim, dtype=object)
        labels[:k] = ["peak_age_%d" % i for i in range(k)]
        labels[k: 2 * k] = [r"$\tau_%d$" % i for i in range(k)]
        labels[2 * k: 3 * k] = ["$Z_%d$" % i for i in range(k)]
        labels[3 * k: 4 * k] = ["$M_{\star,%d}$" % i for i in range(k)]
        labels[-1] = "max_age"
        
        # Plot the walkers
        plt.figure(figsize=(16, 1.5 * ndim))
        for n in range(ndim):
            plt.subplot2grid((ndim, 1), (n, 0))
            plt.plot(sampler.get_chain()[:,:,n], alpha=0.1)
            plt.ylabel(labels[n])
        plt.tight_layout()
        plt.savefig("../walkers.png", bbox_inches="tight", dpi=100)
        plt.close()

        # Flatten the chains, thin them by a factor of 10, and remove the burn-in
        # (first half of the chain)
        chain = sampler.get_chain(flat=True, discard=50, thin=10)

        # Plot marginal posterior distributions
        fig, axes = zeus.cornerplot(chain, labels=labels)
        fig.savefig("../corner_plot.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Get median parameters
        fit_params = np.median(chain, axis=0)

        print("Fit parameters:", fit_params)
    
        # Get the fit
        fit_sed, fit_sfzh = get_spectra(10 ** fit_params[:k] * yr,
                                        fit_params[k: 2 * k],
                                        fit_params[2 * k: 3 * k],
                                        fit_params[3 * k: 4 * k],
                                        10 ** fit_params[-1] * yr,
                                        k)

        print(np.sum(fit_sed.fnu))
    
        # Set up plot
        fig = plt.figure(figsize=(6, 3))
        gs = fig.add_gridspec(nrows=2, ncols=1, wspace=0.0, hspace=0.0,
                              height_ratios=[4, 2])
        ax = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax.grid(True)
        ax1.grid(True)
        
        for i in range(chain.shape[0]):
            sed, _ = get_spectra(10 ** chain[i, :k] * yr,
                                 chain[i, k: 2 * k],
                                 chain[i, 2 * k: 3 * k],
                                 chain[i, 3 * k: 4 * k],
                                 10 ** chain[i, -1] * yr, k)
            ax.loglog(true_sed.lamz, sed._fnu, color="k", alpha=0.3)
    
        ax.loglog(true_sed.lamz, true_sed.fnu, color="g",
                  linestyle="--", label="Truth")
        ax.loglog(true_sed.lamz, fit_sed._fnu, color="c", label="Fit")
        ax1.semilogx(true_sed.lamz,
                     (fit_sed._fnu - true_sed._fnu) / true_sed._fnu)
        ax.set_ylim(10**26., None)

        ax1.set_xlabel('$\lambda / [\AA]$')
        ax.set_ylabel('$F_{\mathrm{nu}}/$ [nJy]')
        ax1.set_ylabel('$F_{\mathrm{Fit}} / F_{\mathrm{Truth}} - 1$')

        handles, leg_labels = ax.get_legend_handles_labels() 

        ax1.legend(handles, leg_labels,
                   loc='upper center', bbox_to_anchor=(0.5, -0.25),
                   fancybox=True, shadow=True, ncol=2)
    
        fig.savefig("../fit_sed_test.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        fig1, ax1 = fit_sfzh.plot(show=False)
        
        fig1.savefig("../fit_SFHZ_test.png")
        plt.close(fig1)

    
