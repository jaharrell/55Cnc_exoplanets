import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, curve_fit
from scipy.stats import norm, trim_mean, chi2, chisquare 
from astropy.timeseries import LombScargle
import time as timer
from samsam import sam, logprior, acf, covis
from kepmodel import rv, tools
import george
from george import kernels
import corner 
from spleaf import term

def scale_time(time):
    # What's a reasonable average timestep and Nyquist frequency?
    nobs = len(time)
    timesteps = np.diff(time)
    meandt = max(time) / (nobs-1)

    tmeandt_10 = trim_mean(timesteps, 0.10)
    tmeandt_20 = trim_mean(timesteps, 0.20)
    meddt = np.median(timesteps)

    Nyq_tmeandt_10 = 0.5 / tmeandt_10
    Nyq_tmeandt_20 = 0.5 / tmeandt_20
    Nyq_meandt = 0.5 / meandt
    Nyq_meddt = 0.5 / meddt

    print("10% trimmed mean dt: ", tmeandt_10, " Nyquist: ", Nyq_tmeandt_10)
    print("20% trimmed mean dt: ", tmeandt_20, " Nyquist: ", Nyq_tmeandt_20)
    print("mean dt: ", meandt, " Nyquist: ", Nyq_meandt)
    print("median dt: ", meddt, " Nyquist: ", Nyq_meddt)
    
    #ptimesteps = histogram(log10.(timesteps), bins=10, legend=false, xlabel=L"\log_{10}(\Delta t)", 
    #             ylabel="Number of timesteps")
    # Shift the median dt correspondingly 
    print("New median dt: ", np.median(np.diff(time/np.median(np.diff(time)))))
    return time/np.median(np.diff(time)), np.median(np.diff(time))

def sort_time(time_array, RV_array, RV_err, inst):
    # Sort by increasing time
    ksort = np.argsort(time_array)
    sorted_time = time_array[ksort]
    sorted_RV = RV_array[ksort]
    sorted_err = RV_err[ksort]
    sorted_inst = inst[ksort]
    return sorted_time, sorted_RV, sorted_err, sorted_inst

def plot_residual(DICT, residual, MODEL, passthrough):
    rv_err = np.sqrt(MODEL.cov.A)
    plt.figure()
    for inst in np.unique(DICT['ins_name']):
        kinst = DICT['ins_name'] == inst
        plt.errorbar(DICT['jd'][kinst],
            residual[kinst],
            yerr=rv_err[kinst],
            fmt='.', rasterized=True, label = inst)
    plt.xlabel('Time [days]')
    plt.ylabel('$v$ [m/s]')
    plt.title("Residual with {} planets removed".format(passthrough))
    plt.legend(loc = 'lower left', fontsize = 'x-small')
    plt.show()
    plt.close()

def plot_GLS(f, power, perc, N_planets_removed):
    # Plot periodogram
    plt.figure(figsize=(10, 5))
    plt.plot(f, power, 'k', lw=1.2, rasterized=True, label = "GLS")
    plt.xlim(np.min(f[1:]), np.max(f))
    plt.vlines(planet_b, min(power), 1.5 * max(power), label = 'Planet b', color = 'b', linestyles = '--')
    plt.vlines(planet_c, min(power), 1.5 * max(power), label = 'Planet c', color = 'g', linestyles = '--')
    #plt.vlines(planet_d, min(power), 1.5 * max(power), label = 'Planet d', color = 'silver', linestyles = '--')
    #plt.vlines(planet_e, min(power), 1.5 * max(power), label = 'Planet e', color = 'y', linestyles = '--')
    plt.vlines(planet_f, min(power), 1.5 * max(power), label = 'Planet f', color = 'm', linestyles = '--')
    #plt.vlines(f_mag, min(power), 1.5 * max(power), label = 'Magentic Cycle', color = 'c', linestyles = '--')
    plt.vlines(f_rot, min(power), 1.5 * max(power), label = 'Stellar Rotation', color = 'chocolate', linestyles = '--')
    plt.vlines((1/29.5), min(power), 1.5 * max(power), label = "Lunar", color = 'blueviolet', linestyles = '-.')
    plt.vlines((1/29.5) + planet_b, min(power), 1.5 * max(power), label = "Lunar + Planet b", color = 'indigo', linestyles = '-.')
    plt.vlines(1./391, min(power), 1.5 * max(power), label = 'P = 391d', color = 'c', linestyles = '--')

    plt.hlines(perc[0], min(f), max(f), label = "0.1% FAP", linestyles = '--', color = "firebrick")
    plt.hlines(perc[1], min(f), max(f), label = "1% FAP", linestyles = '--', color = "maroon")
    plt.hlines(perc[2], min(f), max(f), label = "5% FAP", linestyles = '--', color = "indianred")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [1/day]')
    plt.ylabel('Power')
    plt.title("RV pspec with {} planets removed".format(N_planets_removed))
    plt.legend(loc = 0, fontsize = 'x-small')
    plt.show()
    plt.close()
    
def plot_window(f, window):
    # Plot the spectral window
    plt.figure(figsize=(10, 5))
    plt.plot(f, window, 'k', lw=1.2, rasterized=True, label = "GLS Window") 
    plt.xlim(np.min(f[1:]), np.max(f))
    plt.vlines(planet_b, min(window), 1.5 * max(window), label = 'Planet b', color = 'b', linestyles = '--')
    plt.vlines(planet_c, min(window), 1.5 * max(window), label = 'Planet c', color = 'g', linestyles = '--')
    #plt.vlines(planet_d, min(window), 1.5 * max(window), label = 'Planet d', color = 'silver', linestyles = '--')
    #plt.vlines(planet_e, min(window), 1.5 * max(window), label = 'Planet e', color = 'y', linestyles = '--')
    plt.vlines(planet_f, min(window), 1.5 * max(window), label = 'Planet f', color = 'm', linestyles = '--')
    plt.vlines(f_mag, min(window), 1.5 * max(window), label = 'Magentic Cycle', color = 'c', linestyles = '--')
    plt.vlines(f_rot, min(window), 1.5 * max(window), label = 'Stellar Rotation', color = 'chocolate', linestyles = '--')
    plt.vlines((1/29.5), min(window), 1.5 * max(window), label = "Lunar", color = 'blueviolet', linestyles = '-.')
    plt.vlines((1/29.5) + planet_b, min(window), 1.5 * max(window), label = "Lunar + Planet b", color = 'indigo', linestyles = '-.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency [cycles/day]')
    plt.ylabel('Power')
    plt.title("GLS Window")
    plt.legend(loc = 0)
    plt.show()
    plt.close()
    
    
def plot_GP(gp, param_vec, time, RV, RVerr):
    gp.set_parameter_vector(param_vec) # Update GP
    pred, pred_var = gp.predict(RV, time, return_var = True)
    #gp_sample = gp.sample_conditional(RV, time)

    plt.figure(figsize = (10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharey = ax1, sharex = ax1)
    
    ax1.fill_between(time, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                    color="k", alpha=0.2)
    ax1.plot(time, pred, "g", lw=1.3, alpha=0.5)
    ax1.errorbar(time, RV, yerr = RVerr, fmt = ".k", capsize = 0)
    #ax1.plot(time, gp_sample, "-g")
    ax1.set_xlabel("Time [days]")
    ax2.errorbar(time, RV - pred, yerr = RVerr, fmt = ".k", capsize = 0.0)
    ax2.set_xlabel("Time [days]")
    plt.title("55Cnc GP Detrended")
    plt.show()
    plt.close()
    return time, pred
    

def GLS_bootstrap(t, f, rv):
    # Initialize arrays 
    N_bootstrap = 10000 
    S_total = np.zeros((len(f[1:]), N_bootstrap))
    percentiles = np.zeros(3)
    maxima = np.zeros(N_bootstrap)
    start = timer.time()
    # Bootstrap loop
    for i in range(N_bootstrap):
        sample = np.random.choice(rv, len(rv), replace = True) # Take a random sample of the data
        GLS_object = LombScargle(t, sample, normalization = 'psd') 
        S_total[:, i] = normalize(sample, f[1:], GLS_object.power(f[1:])) # Exclude 0 frequency to avoid divide by zero
        maxima[i] = np.max(S_total[:, i])
        if (i/N_bootstrap) % 0.1 == 0:
            print(i/N_bootstrap) # Completion percent
    end = timer.time()
    print("Bootstrap took", end - start, "s for", N_bootstrap, "iterations")

    # Compute percentiles
    percentiles = np.percentile(maxima, [95.0, 99.0, 99.9]) 
    return percentiles

def GLS_calc(time, ts, bootstrap = False):
        RR = 1./(max(time)-time[0])
        delta_f = RR/2
        f_N = 1./(2*np.median(np.diff(time))) # Nyquist frequency
        f = np.arange(0, f_N, delta_f)
    
        GLS_object = LombScargle(time, ts, normalization = 'psd') # Pre-plan the periodogram
        GLS = GLS_object.power(f[1:]) # Exclude 0 frequency to avoid divide by zero
        if bootstrap:
            percentiles = GLS_bootstrap(time, f, ts)
        # LS taper
        GLS_taper = np.ones(len(time))
        GLS_window = LombScargle(time, GLS_taper, normalization = 'psd').power(f[1:])

        #window = np.where(f <= W)
        #in_window = sum(GLS_window[window])
        #total = sum(GLS_window)
        #GLSconcen = in_window/total # sum(GLS_window_nfft[np.where(f <= W)])/sum(GLS_window_nfft)
        #print("Spectral concentration for Lomb-Scargle: ", GLSconcen)
        
        if bootstrap:
            return f[1:], normalize(ts, f[1:], GLS), GLS_window, percentiles
        else:
            return f[1:], normalize(ts, f[1:], GLS), GLS_window

def normalize(ts, f_array, spectrum, return_int = False):
        var = np.var(ts)
        df = f_array[2] - f_array[1]
        integral = (sum(spectrum) * df)
        
        if return_int:
                return spectrum*(var/integral), integral
        else:
                return spectrum*(var/integral)

# ====================== Detrending functions ====================== 
# Polynomial detrending 
def detrend(time, RV, deg):
    coeff = np.polyfit(time, RV, deg)
    if deg == 2:
        trend = coeff[0]*time**2 + coeff[1]*time**1 + coeff[2]
    elif deg == 3:
        trend = coeff[0]*time**3 + coeff[1]*time**2 + coeff[2]*time**1 + coeff[3]
    elif deg == 4:
        trend = coeff[0]*time**4 + coeff[1]*time**3 + coeff[2]*time**2 + coeff[3]*time**1 + coeff[4]
    rv_detrended = RV - trend
    
    print(coeff)
    return rv_detrended, trend 

# Sinusoidal fit 
def sinusoid(t, f, A, phi):
    s = A*np.sin(2*np.pi*t*f + phi)
    return s

# Lambda function: Gaussian filter. t is the array of observation times,
# tc is the time at which you want to center your Gaussian, and sc is
# the scale, or standard deviation.
Gausskern = lambda t, tc, sc: np.exp(-((t-tc)**2. / (2.*sc**2.)))

'''Function that smooths the time series with Gaussian averaging'''
# Inputs: observation times (Julian dates or such), y-values at each time,
# smoothing scale length
def Gsmooth(times, yvals, scale):
    nobs = len(times) # Number of observations
    ysmooth = np.zeros(nobs)
    for i in range(nobs):
        Gk = Gausskern(times, times[i], scale)
        # Gausskern = np.exp(-((times-times[i])**2. / (2.*scale**2.)))
        Gausssum = np.sum(Gk)
        ysmooth[i] = np.sum(yvals*Gk) / Gausssum
    return(ysmooth)


# ========================= GP Functions =========================

# Function that takes the two parameter vectors and turns them
# into one master parameter vector; also sanity checks that the
# periods and metrics are the same for the quasiperiodic kernels
def make_par_vec(RVpars, Spars):
    RV_amp = RVpars[0]
    RV_metric = RVpars[1]
    RV_gamma = RVpars[2]
    RV_period = RVpars[3]
    S_amp = Spars[0]
    S_metric = Spars[1]
    S_gamma = Spars[2]
    S_period = Spars[3]    
    assert RV_period == S_period, "Error: RV and S-index should have same period"
    assert RV_metric == S_metric, "Error: RV and S-index should have same decorrelation timescale"
    master_pars = [RV_period, RV_metric, RV_amp, RV_gamma, \
                   S_amp, S_gamma]
    return master_pars

def split_par_vec(par_vec):
    period = par_vec[0]
    metric = par_vec[1]
    RV_amp = par_vec[2]
    RV_gamma = par_vec[3]
    S_amp = par_vec[4]
    S_gamma = par_vec[5]
    RV_pars = [RV_amp, metric, RV_gamma, period]
    S_pars = [S_amp, metric, S_gamma, period]
    return RV_pars, S_pars

def split_kep_gp(all_params):
    kep_params =  all_params[0:len(all_params)-4]
    GP_params = all_params[len(all_params)-4:] # Last 4 should be GP params
    return kep_params, GP_params

def combine_kep_GP_params(kep_params, GP_params):
    master_params = np.zeros(len(kep_params)+len(GP_params))
    master_params[0:len(master_params)-4] = kep_params
    master_params[len(master_params)-4:] = GP_params
    return master_params

def second_deriv(x, y):
    n = len(x)
    h = np.mean(np.diff(x))      
    d2y_dx2 = -999.0*np.ones(n-2)
    for i in range(1, n-1):
        d2y_dx2[i-1] = (y[i+1] - 2*y[i] + y[i-1])/(h**2)
    return d2y_dx2

# Basic values for plotting, etc
f_mag = 1.0/(10.5*365.25) #This equals 1/3835.125 (Bourrier value: 1.0/3822.4)
f_rot = 1.0/(38.8) # From Bourrier et al. 2018
planet_b = 1.0/14.65314 # (Bourrier value: 1.0/14.6516)          
planet_c = 1.0/44.373 #(Bourrier value: 1.0/44.3989)             
planet_d = 1.0/4867  #(Bourrier value: 1.0/5574.2)               
planet_e = 1.0/0.7365478 #(Bourrier value: 1.0/0.73654737) # has transit confirmation
planet_f = 1.0/260.91 #(Bourrier value: 1.0/259.88)              
lunar = 1.0/29.53