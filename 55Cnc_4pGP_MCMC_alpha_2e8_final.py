import numpy as np
from scipy.optimize import minimize, Bounds, curve_fit
from samsam import sam, logprior, acf, covis
from kepmodel import rv, tools
from spleaf import term
import george
from george import kernels
# ======================================================== Read in data ========================================================

# Read in RV and S-index
inst_Bourrier  = np.loadtxt("/home/3363/55Cnc/data/Bourrier_RV.txt", dtype = str, usecols = (5), skiprows = 43)
t_Bourrier, RV_Bourrier, RVerr_Bourrier, S_Bourrier, Serr_Bourrier = np.loadtxt("/home/3363/55Cnc/data/Bourrier_RV.txt", usecols = (0, 1, 2, 3, 4), skiprows = 43, unpack = True ) # BINNED, TRENDED 
t_Bourrier = t_Bourrier + 2400000.0 # Shift to BJD

# Sort by time index
def sort_time(time_array, RV_array, RV_err, inst):
    # Sort by increasing time
    ksort = np.argsort(time_array)
    sorted_time = time_array[ksort]
    sorted_RV = RV_array[ksort]
    sorted_err = RV_err[ksort]
    sorted_inst = inst[ksort]
    return sorted_time, sorted_RV, sorted_err, sorted_inst

t_Bourrier, RV_Bourrier, RVerr_Bourrier, inst_Bourrier = sort_time(t_Bourrier, RV_Bourrier, RVerr_Bourrier, inst_Bourrier)
t_Bourrier_S, S_Bourrier, Serr_Bourrier, inst_Bourrier_S = sort_time(t_Bourrier, S_Bourrier, Serr_Bourrier, inst_Bourrier)

# Define signal frequencies as informed by prior literature 
f_mag = 1.0/(10.5*365.25) #This equals 1/3835.125 (Bourrier value: 1.0/3822.4)
f_rot = 1.0/(38.8) # From Bourrier et al. 2018
planet_b = 1.0/14.65314 # (Bourrier value: 1.0/14.6516)          
planet_c = 1.0/44.373 #(Bourrier value: 1.0/44.3989)             
planet_d = 1.0/4867  #(Bourrier value: 1.0/5574.2)               
planet_e = 1.0/0.7365478 #(Bourrier value: 1.0/0.73654737)       
planet_f = 1.0/260.91 #(Bourrier value: 1.0/259.88)              
lunar = 1.0/29.53
# ==============================================================================================================================


# ======================================================== Construct the Keplerian Model  ========================================================
# drift power
dpow = 0 # Not used here

instruments = inst_Bourrier 

fit_method = 'L-BFGS-B' 
fit_options = {
    'maxiter': 1000,
    'maxcor': 50
}

fit_ecc = True
fap_max = 1e-3 # FAP over this value terminates the agnostic DACE search (not used here)
zeropad = 2.0

rv_model = rv.RvModel(t_Bourrier-t_Bourrier[0], RV_Bourrier, err = term.Error(RVerr_Bourrier))

# Construct a dictionary for the RV data
rv_dict = {}
rv_dict['jd'] = t_Bourrier-t_Bourrier[0] 
rv_dict['rv'] = RV_Bourrier
rv_dict['rv_err'] = RVerr_Bourrier
rv_dict['ins_name'] = instruments

N = len(rv_dict['jd']) 
RR = 1./(max(rv_dict['jd'])-rv_dict['jd'][0]) # Rayleigh Resolution, to be used in frequency analysis

# Add linear parameters
for inst in np.unique(instruments):
    rv_model.add_lin(1.0*(rv_dict['ins_name']==inst), f'offset_inst_{inst}') 

rv_model.fit(method=fit_method, options=fit_options)
rv_model.show_param();

rv_err = np.sqrt(rv_model.cov.A)
signals = [planet_b, planet_c, planet_f, planet_e]
for passthrough in range(len(signals)):
	rv_model.add_keplerian_from_period((1.0/signals[passthrough]), fit = True) 
	rv_model.set_keplerian_param(f'{rv_model.nkep-1}', param=['P', 'la0', 'K', 'e', 'w']) # First argument is name
	if not fit_ecc:
		rv_model.set_param(np.zeros(2), rv_model.fit_param[-2:])
		rv_model.fit_param = rv_model.fit_param[:-2]
	# Global fit of the model
	rv_model.fit(method=fit_method, options=fit_options)
	rv_model.show_param();
# ==============================================================================================================================


# ======================================================== Construct the GP Model  ========================================================
time = rv_dict['jd'] - rv_dict['jd'][0]; RV = RV_Bourrier; RVerr = RVerr_Bourrier 
unif_time = np.linspace(time[0], np.max(time), round(len(time)/3)) 
kep_model = rv_model
N_param = len(rv_model.get_param())

metric_guess = 17.0 
gamma_guess = 1.0 
period_guess = np.exp(8.55) 

# Set reasonable boundaries for each hyperparameter
# Parameter order: ln(amplitude), ln(metric^2), gamma, ln(period)
GP_lower_bounds = [0.5, np.log(2000.0**2), 0.01, np.log(1500.0)]  
GP_upper_bounds = [5.8, 22.0, 10.0, 8.8] 
model_values = np.array(kep_model.get_param())
model_errors = np.array(kep_model.get_param_error()[1])

# Fix the nan in uncertanties to Bourrier published value
model_errors[N_param-5] = 1.3*10**(-6) 

# Vastly increasing parameter error bars to allow GP to explore more space
# (Only affects instrumental offsets, other bounds fixed below)
kep_lower_bounds = model_values - 1000.0*model_errors 
kep_upper_bounds = model_values + 1000.0*model_errors

for p in range(len(signals)):
    kep_lower_bounds[7 + int(5*p) + 1] = 0.0 # Mean longitude
    kep_lower_bounds[7 + int(5*p) + 3] = 0.0 # eccentricity
    kep_lower_bounds[7 + int(5*p) + 4] = 0.0 # argument of periastron
    
    kep_upper_bounds[7 + int(5*p) + 1] = 360.0 # Mean longitude
    kep_upper_bounds[7 + int(5*p) + 3] = 0.7 # eccentricity
    kep_upper_bounds[7 + int(5*p) + 4] = 360.0 # argument of periastron

# Fix amplitude bounds 
kep_lower_bounds[16-7] = 50.0; kep_upper_bounds[16-7] = 80.0 # Amplitude of planet b
kep_lower_bounds[21-7] = 2.0; kep_upper_bounds[21-7] = 20.0 # Amplitude of planet c
kep_lower_bounds[26-7] = 2.0; kep_upper_bounds[26-7] = 20.0 # Amplitude of planet f
kep_lower_bounds[26-9] = 200.0; kep_upper_bounds[26-9] = 300.0 # Period of planet f
kep_lower_bounds[31-7] = 2.0; kep_upper_bounds[31-7] = 20.0 # Amplitude of planet e

def combine_kep_GP_params(kep_params, GP_params):
    master_params = np.zeros(len(kep_params)+len(GP_params))
    master_params[0:len(master_params)-4] = kep_params
    master_params[len(master_params)-4:] = GP_params
    return master_params

par_bounds = Bounds(combine_kep_GP_params(kep_lower_bounds, GP_lower_bounds), combine_kep_GP_params(kep_upper_bounds, GP_upper_bounds))
    
k_exp2 = np.std(kep_model.residuals()) * kernels.ExpSquaredKernel(metric = metric_guess) 
k_per = kernels.ExpSine2Kernel(gamma = gamma_guess, log_period = np.log(period_guess))
k_mag = k_exp2 * k_per

# The below guess parameters are prior MCMC/Nelder-Mead results
guess_pars_no_jitter = [ 2.74593149e+04,  2.74735295e+04,  2.84020578e+04, -3.49661346e+01,
        8.70454913e+00,  2.74434876e+04, -2.25657971e+04,  1.46515426e+01,
        5.70301649e+00,  7.12587763e+01,  8.19261292e-04,  1.07291191e+01,
        4.44028319e+01,  1.75853994e-01,  9.72565737e+00,  3.52274641e-02,
        6.53886270e+00,  2.59836810e+02,  5.09250798e+00,  5.19292021e+00,
        2.36270506e-01,  5.35101354e+00,  7.36548143e-01,  1.94961634e+00,
        6.02062974e+00,  3.28863413e-02,  2.06037193e+00,  3.88512192e+00,
        1.68154242e+01,  3.51459936e+00,  8.48215260e+00]
        
for i in range(len(kep_lower_bounds)):
    print("Min | Initial Value | Max : ", kep_lower_bounds[i], "|", guess_pars_no_jitter[i], "|", kep_upper_bounds[i])        

gp_irregular = george.GP(k_mag, mean = np.mean(kep_model.residuals()), fit_kernel = True)
gp_irregular.set_parameter_vector(guess_pars_no_jitter[len(guess_pars_no_jitter)-4:]) 
gp_irregular.compute(time, RVerr)
# ==============================================================================================================================


# ======================================================== Define the Objective Function and Optimize ========================================================
alpha = 2.0e8 
def split_kep_gp(all_params):
    kep_params =  all_params[0:len(all_params)-4]
    GP_params = all_params[len(all_params)-4:] # Last 4 should be GP params
    return kep_params, GP_params

def second_deriv(x, y):
    n = len(x)
    h = np.mean(np.diff(x))      
    d2y_dx2 = -999.0*np.ones(n-2)
    for i in range(1, n-1):
        d2y_dx2[i-1] = (y[i+1] - 2*y[i] + y[i-1])/(h**2)
    return d2y_dx2

def penalized_NLL(all_params):
    kep_params, GP_params = split_kep_gp(all_params)
    kep_model.set_param(kep_params)
    kepres = kep_model.residuals() # Update residual 
    gp_irregular.set_parameter_vector(GP_params)    

    # Compute mean GP prediction
    GP_pred_unif, pred_var_unif = gp_irregular.predict(kepres, unif_time, return_var = True) 
    GP_pred_irregular, pred_var_irregular = gp_irregular.predict(kepres, time, return_var = True)
    
    deriv_sum = (second_deriv(unif_time, GP_pred_unif)**2).sum()
    obj = -0.5*((kepres-GP_pred_irregular)**2/RVerr_Bourrier**2).sum() - 0.5*alpha*deriv_sum
    return -obj if np.isfinite(obj) else 1e25

def penalized_NLL_PRINTMODE(all_params): # Same as above, just prints out the values of the two terms in the obj func
    kep_params, GP_params = split_kep_gp(all_params)
    kep_model.set_param(kep_params)
    kepres = kep_model.residuals() # Update residual 
    gp_irregular.set_parameter_vector(GP_params)    

    # Compute mean GP prediction
    GP_pred_unif, pred_var_unif = gp_irregular.predict(kepres, unif_time, return_var = True) 
    GP_pred_irregular, pred_var_irregular = gp_irregular.predict(kepres, time, return_var = True)
    
    deriv_sum = (second_deriv(unif_time, GP_pred_unif)**2).sum()
    obj = -0.5*((kepres-GP_pred_irregular)**2/RVerr_Bourrier**2).sum() - 0.5*alpha*deriv_sum
    
    print("LSQ term = ", -0.5*((kepres-GP_pred_irregular)**2/RVerr_Bourrier**2).sum())
    print("Curvature term = ", - 0.5*alpha*deriv_sum)
    print("alpha = ", alpha)
    return -obj if np.isfinite(obj) else 1e25

print("starting optimization")
# Starting from last results
params = np.loadtxt('/home/3363/55Cnc/text_outputs/4pGP/55Cnc_4pGP_NM_param_alpha_1000000000.txt')
print(penalized_NLL_PRINTMODE(params))
print("LAST RUN GP params: ")
print("Fitted GP period [days]: ", np.exp(params[len(params)-1]))
print("GP decor. timescale [days]: ", np.sqrt(np.exp(params[len(params)-3])) ) 
print("Fitted GP amplitude [m/s]: ", np.exp(params[len(params)-4]))

nsamples = 400000  # MCMC iterations

lower_param_bound = combine_kep_GP_params(kep_lower_bounds, GP_lower_bounds)
upper_param_bound = combine_kep_GP_params(kep_upper_bounds, GP_upper_bounds)

def lprior_kep_GP(x):
    lp = 0
    for i in range(len(x)): 
        lp += logprior.uniform(x[i], a = lower_param_bound[i], 
                                 b = upper_param_bound[i]) # Same bounds as Nelder-Mead
    return lp    

def lprob_kep_GP(x):
    try:
        lp = lprior_kep_GP(x)
        if not (lp > -np.inf):
            return (-np.inf)
        ll = -penalized_NLL(x) 
        if np.isnan(ll):
            return(-np.inf)
        return(lp + ll)
    except:
        return (-np.inf)

samples, diagnostics = sam(params, lprob_kep_GP, nsamples=nsamples, print_level = 1, print_interval = 20000)

np.savetxt("/home/3363/55Cnc/text_outputs/4pGP/MCMC/55Cnc_4pGP_MCMC_chains_alpha_{0:.0f}_newamp.txt".format(alpha), samples.flatten(), header = f"N_MCMC = {nsamples}, alpha = {alpha}, GP_lower_bounds = {GP_lower_bounds}, GP_upper_bounds = {GP_upper_bounds}")

np.savetxt("/home/3363/55Cnc/text_outputs/4pGP/MCMC/55Cnc_4pGP_MCMC_param_alpha_{0:.0f}_newamp.txt".format(alpha), diagnostics['mu'], header = f"N_MCMC = {nsamples}, alpha = {alpha}, GP_lower_bounds = {GP_lower_bounds}, GP_upper_bounds = {GP_upper_bounds}")

np.savetxt("/home/3363/55Cnc/text_outputs/4pGP/MCMC/55Cnc_4pGP_MCMC_acceptrate_alpha_{0:.0f}_newamp.txt".format(alpha), diagnostics['alpha'], header = f"N_MCMC = {nsamples}, alpha = {alpha}, GP_lower_bounds = {GP_lower_bounds}, GP_upper_bounds = {GP_upper_bounds}")

np.savetxt("/home/3363/55Cnc/text_outputs/4pGP/MCMC/55Cnc_4pGP_MCMC_logprob_alpha_{0:.0f}_newamp.txt".format(alpha), diagnostics['logprob'], header = f"N_MCMC = {nsamples}, alpha = {alpha}, GP_lower_bounds = {GP_lower_bounds}, GP_upper_bounds = {GP_upper_bounds}")

