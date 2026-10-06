import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import emcee
import getdist
import time
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.linalg as la
from matplotlib import rc
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from getdist import plots, MCSamples

z_data_sn, mu_sn, mu_err_sn = np.loadtxt("data1/Pantheon+SH0ES.dat", usecols = (2,10,11), unpack = True, skiprows = 1)

cov_data = np.loadtxt("data1/Pantheon+SH0ES_STAT+SYS.cov")
cov_mat = cov_data[1:].reshape(1701,1701)
inverse_covar = la.inv(cov_mat)

def wd(z, params):
	return -1.0



def equation(z, variable, params):

	od, H, dl = variable

	od0, H0 = params
	
	dotH = (-3 * (wd(z, params) * od + 1))
	
	edq = 1 / (1+z) * (3 * od * (1 + wd(z, params)) + dotH * od)
	
	edH = (3 / (2 * (1+z))) * H * (wd(z, params) * od + 1)

	eqdl = 1 / (1+z) * dl + (1 / H) * 2.99792458e5 * (1 + z)

	return [edq, edH, eqdl] 



def mu_model(z, params): 
	
	od0, H0 = params
	
	mu_values = np.zeros(len(z_data_sn))
	sol = solve_ivp(lambda t, y: equation(t, y, params), [0,3], [od0, H0, 0], t_eval = np.unique(z_data_sn), method = 'RK45')
	dl_sol = sol.y[2]
	dl_val = interp1d(sol.t, dl_sol, kind = "linear", fill_value = "extrapolate")
	mu_val = 5 * np.log10(dl_val(z)) + 25

	return mu_val



def chisq(D, T, err):
	
	diff = D - T
	diffT = diff.T
	chisq = np.dot(diffT, np.dot(err, diff))
	
	return chisq 



def log_prior(params): 
	
	od0, H0 = params
	
	if not (40 < H0 < 99): 
		return -np.inf
	
	if not (0.5 < od0 < 1): 
		return -np.inf

	return 0



def log_prob(params): 
	
	prior = log_prior(params)
	
	if prior == -np.inf:
		return -np.inf

	mu = mu_model(z_data_sn, params)

	if np.any(np.isinf(mu)):
		return -np.inf

	chisq_value = chisq(mu_sn, mu, inverse_covar)
	
	return -0.5 * chisq_value



def aic(log_likelihood, ndim):
	return -2 * log_likelihood + 2 * ndim



def bic(log_likelihood, ndim, ndata): 
	return -2 * log_likelihood + ndim * np.log(ndata)



if __name__ == "__main__" :
	
	nwalker = 30
	ndim = 2
	niter = 50000

	p0 = np.random.uniform(low = [0.5, 40.], high = [1, 99], size = (nwalker, ndim))
	
	ncpu = cpu_count()
	
	print(f"{ncpu} CPUs")
	
	start_time = time.time()
	
	with Pool(processes = 6) as pool:
		sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob, pool = pool)
		sampler.run_mcmc(p0, niter, progress = True)
	
	elapsed_time = time.time() - start_time
	
	print(f"\n Execution Time: {elapsed_time:.2f} seconds")
	
	try:
		tau = sampler.get_autocorr_time(quiet = True)
		max_tau = np.max(tau)
		
		print(f"Autocorrelation times (tau): Omega_d = {tau[0]:.1f}, H0 = {tau[1]:.1f}")
		
		print(f"Convergence Ratio (Niter/max tau): {niter / max_tau:.1f}x")
	
	except Exception:
		print("The autocorrelation time couldn't be computed because the chain is too short.")

	
	dis = 5000
	th = 20
	
	chains = sampler.get_chain(flat = True, discard = dis, thin = th)
	
	samples = sampler.get_chain(discard = dis, thin = th, flat = True)
	
	name = ['od0', 'H0']
	labels1 = [r'\Omega_D', r'H0']
	
	sample2 = MCSamples(samples = samples, names = name, labels = labels1)
	
	log_likelihoods = sampler.get_log_prob()
	
	log_likelihood = np.max(log_likelihoods)
	
	print("Log likelihood is", log_likelihood)
	
	print("AIX values:", aic(log_likelihood, ndim))
	
	aic_value = aic(log_likelihood, ndim)
	
	bic_value = bic(log_likelihood, ndim, len(z_data_sn))
	
	names = ['od0', 'H0']
	labels = [r'\Omega_d', r'H0']
	
	sample2 = MCSamples(samples = chains, names = names, labels = labels)
	
	od_mc, H0_mc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), 
								zip(*np.percentile(samples, [16, 50, 84], 
												   axis = 0)))
	print(od_mc, H0_mc)
	
	flat_samples = sampler.get_chain(discard = dis, thin = 2, flat = True)
	
	label = [r'\Omega_d', r'H0']

	with open("SN_statistics_basic.txt", "w") as file: 
		for i in range(ndim): 
			mcmc = np.percentile(flat_samples[:,i] , [16, 50, 84])
			q = np.diff(mcmc)
			txt = r'\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}'
			txt = txt.format(mcmc[1], q[0], q[1], label[i])
			file.write(txt + "\n")
		file.write(f"AIC value = {aic_value:.2f}\n")
		file.write(f"BIC value = {bic_value:.2f}\n")

	g = plots.get_subplot_plotter(width_inch = 6)
	g.settings.figure_legend_frame = True
	g.settings.alpha_filled_add = 0.6
	g.settings.title_limit_fontsize = 14
	g.settings.axes_labelsize = 12
	g.settings.legend_fontsize = 10
	g.settings.colorbar_axes_fontsize = 10
	g.triangle_plot(sample2, ['od0', 'H0'], filled = True, contour_colors = ['red'], title_limit = 1)
	
	g.export("SN_chain_git.pdf")	

