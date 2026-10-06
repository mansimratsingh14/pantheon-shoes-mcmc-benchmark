import glob
import pandas as pd
import numpy as np
import sncosmo as sn
from astropy.table import Table

def analyze_lightcurves(file):
	with open(file) as f: 
		for line in f: 
			if "PEAKMJD" in line:
				peak_mjd = float(line.split()[1])
			elif "SNID" in line: 
				snid = int(line.split()[1])
			elif "REDSHIFT_CMB" in line: 
				redshift_cmb = float(line.split()[1])
				redshift_cmb_err = float(line.split()[3])
			elif "NOBS" in line:
				nobs = int(line.split()[1])
			elif "RA" in line: 
				ra = float(line.split()[1])
			elif "DEC" in line: 
				dec = float(line.split()[1])
			elif "MWEBV" in line: 
				mwebv = float(line.split()[1])
			elif "VPEC" in line: 
				vpec = float(line.split()[1])
				vpec_err = float(line.split()[3])
			elif "REDSHIFT_HELIO" in line: 
				redshift_helio = float(line.split()[1])
				redshift_helio_err = float(line.split()[3])
			elif "HOSTGAL_PHOTOZ" in line: 
				hostgal_photoz = float(line.split()[1])
				hostgal_photoz_err = float(line.split()[3])
			elif "HOSTGAL_SPECZ" in line: 
				hostgal_specz = float(line.split()[1])
				hostgal_specz_err = float(line.split()[3])
			elif "HOSTGAL_SNSEP" in line:
				hostgal_snsep = float(line.split()[1])
			elif "HOSTGAL_LOGMASS" in line: 
				hostgal_logmass = float(line.split()[1])
				hostgal_logmass_err = float(line.split()[3])
			elif "HOSTGAL_MAG" in line: 
				parts = line.split(":")
				words = parts[0].split()
				hostgal_mag = parts[1].split()
			elif "HOSTGAL_SB_FLUXCAL" in line: 
				parts = line.split(":")
				words = parts[0].split()
				hostgal_sb_fluxcal = parts[1].split()

	with open(file) as f: 
		for i , line in enumerate(f): 
			if line.startswith("VARLIST:"): 
				data_start = i + 1
				break

	df = pd.read_csv(file, sep = r"\s+", header = None, skiprows = data_start, nrows = nobs)

	df = df.iloc[:,1:]

	df.columns = ["MJD", "BAND", "FIELD", "FLUXCAL", "FLUXCALERR", "SIM_MAGOBS", 
		"ZPPFLUX", "PSF", "SKYSIG", "GAIN", "PHOTFLAG", "PHOTPROB"]

	df = df.drop(columns = "FIELD")

	nobs_z = (df["BAND"] == "z").sum()
	nobs_i = (df["BAND"] == "i").sum()
	nobs_r = (df["BAND"] == "r").sum()
	nobs_g = (df["BAND"] == "g").sum()

	mjd_max = df["MJD"].max()
	mjd_min = df["MJD"].min()

	model = sn.Model(source = "salt2")

	band_map = {"g": "desg", 
				"r": "desr", 
				"i": "desi", 
				"z": "desz"}

	df["BAND"] = df["BAND"].map(band_map)

	lc = df[["MJD", "BAND", "FLUXCAL", "FLUXCALERR"]].copy()

	lc = lc.rename(columns = {"MJD" : "time", 
								"BAND": "band",
								"FLUXCAL": "flux",
								"FLUXCALERR":"fluxerr"})


	lc["zp"] = 27.5
	lc["zpsys"] = "ab" 

	model.set(z = redshift_cmb)
	lc = Table.from_pandas(lc)

	try:
		results, fitted_model = sn.fit_lc(lc,
										model, 
										["t0", "x1", "c"])

		t0 = results.parameters[results.param_names.index("t0")]
		x1 = results.parameters[results.param_names.index("x1")]
		c = results.parameters[results.param_names.index("c")]

		stretch_factor = 0.98 + 0.091 * x1 + 0.003 * x1**2 - 0.00075 * x1**3

		dm15 = 1.09 - 0.161 * x1 + 0.0013 * x1**2 - 0.000130 * x1**3

		df["PHASE"] = df["MJD"] - t0
		t0_err = results.errors["t0"]
		x1_err = results.errors["x1"]
		c_err = results.errors["c"]

	except Exception as exp:
		
		print(f"The SALT2 fit failed for {snid} due to: {exp}")
		return None
	
	
	t_template = 15.0

	t_decay = t_template * stretch_factor 

	return { "SNID": snid, 
				"Redshift CMB": redshift_cmb, 
				"Stretch factor": stretch_factor, 
				"Phillips parameter": dm15, 
				"Decay time": t_decay, 
				"x1": x1, 
				"x1_err": x1_err,
				"t0": t0,
				"t0_err": t0_err,
				"c": c,
				"c_err": c_err,
				"Fit success": results.success,
				"MJD MIN": mjd_min, 
				"MJD MAX": mjd_max
			}


files = glob.glob("data/*.dat")

results = []

for file in files:
	try:
		result = analyze_lightcurves(file)
		
		if result == None:
			continue
		
		results.append(result)
	except pd.errors.EmptyDataError:
		print(f"Skipped empty files : {file}")

results_df = pd.DataFrame(results)

filter_cuts = ((-3.0 < results_df["x1"]) & (results_df["x1"] <= +3.0) & 
				(-0.3 < results_df["c"]) & (results_df["c"] < +0.3) & 
				(results_df["x1_err"] <= 1.0) & 
				(results_df["c_err"] < 0.1) & 
				(results_df["t0_err"] < 2.0))

fresults_df = results_df[filter_cuts].copy()

outliers = fresults_df[
			(fresults_df["t0"] < fresults_df["MJD MIN"] - 15) | 
			(fresults_df["t0"] > fresults_df["MJD MAX"] + 15)
			]


print(f"Number of outliers: {len(outliers)}")

if len(outliers) > 0:
		print(outliers[["SNID", "t0", "MJD MIN", "MJD MAX", "x1", "c"]])

final_results_df = fresults_df.drop(outliers.index).copy()
final_results_df.to_csv("des_results.csv", index = False)

print(final_results_df.shape)
print(final_results_df)


































































































