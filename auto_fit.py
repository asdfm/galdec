import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse, build_ellipse_model
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.table import Table
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import scipy.special as sc
from scipy.interpolate import interp1d
import corner
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from dustmaps.csfd import CSFDQuery
from lmfit import Model
from astropy.convolution.kernels import Gaussian1DKernel
# from plot_phot_profile import *

import threading
from multiprocessing.pool import Pool

from IPython.core.debugger import set_trace

from scipy import interpolate
from scipy import signal
import photutils
from scipy.signal import peak_widths
from scipy.signal import find_peaks
# from skimage.feature import peak_local_max
from scipy.signal import argrelextrema
from astropy.table import QTable
import matplotlib.image as mpimg
from copy import deepcopy

'''
run_emcee_fit() parameters:

params: parameters undergoing minimisation, see class lmfit.Parameters.

Example of creation of params from components (see description below):
###
params = Parameters()
params.add('mu_0'+str(0), value=components[0]['mu_0'], min=0)
for i in range(0, len(components)):
    vary = False
    if i in [0,1,2,3,4]:
        vary = True
    else:
        vary = False

    if components[i]['type'] == 'sersic':
        params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10, max=30, vary=vary)
        params.add('r_0'+str(i), value=components[i]['r_0'], min=1e-3, max=10, vary=vary)
        params.add('n'+str(i), value=components[i]['n'], min=1e-3, max=10, vary=vary)
    if components[i]['type'] == 'psf':
        params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=0, max=30, vary=vary)
###
        
components: currently only with type:"sersic", exp disc is type:"sersic" with vary=False for n=1 (see lmfit.Parameters).
###
components = {0:{'type':'sersic', 'mu_0':21.86, 'r_0':2.66, 'n':1.48},
              1:{'type':'sersic', 'mu_0':20.57, 'r_0':1.31, 'n':0.32},
              2:{'type':'sersic', 'mu_0':19.32, 'r_0':0.174, 'n':0.961},
              3:{'type':'sersic', 'mu_0':18.25, 'r_0':0.0257, 'n':0.823}}
###
              
iso_table: array with photutils.isophote.Ellipse results in the form of astropy.Table

pixsize: size of a pixel in arcsec.

scale: kpc per arcsec.

psf: 1D array with normalized psf in pixel coordinates.

zp: zero-point magnitude.

cornerplot: save emcee cornerplots if True.
'''
def fill_components(params_file):
    components = {}
    params_table = fits.open(params_file)[1].data

    comp = "sersic"
    idxs = (params_table["comps_names"][0] == comp).nonzero()[0]
    for i in range(len(idxs)):
        components[idxs[i]] = {'type':"sersic", 'mu_0':params_table["sersic_mu_0"][0][i], 'r_0':params_table["sersic_r_0"][0][i], 'n':params_table["sersic_n"][0][i]}
    
    comp = "exp_disk"
    idxs = (params_table["comps_names"][0] == comp).nonzero()[0]
    for i in range(len(idxs)):
        components[idxs[i]] = {'type':"exp_disk", 'mu_0':params_table["exp_disk_mu_0"][0][i], 'h':params_table["exp_disk_h"][0][i]}

    comp = "bar"
    idxs = (params_table["comps_names"][0] == comp).nonzero()[0]
    for i in range(len(idxs)):
        components[idxs[i]] = {'type':"bar", 'mu_0':params_table["bar_mu_0"][0][i], 'r':params_table["bar_r"][0][i], 'h':params_table["bar_h"][0][i]}
    
    comp = "ring"
    idxs = (params_table["comps_names"][0] == comp).nonzero()[0]
    for i in range(len(idxs)):
        components[idxs[i]] = {'type':"ring", 'mu_0':params_table["ring_mu_0"][0][i], 'r':params_table["ring_r"][0][i], 'w':params_table["ring_w"][0][i]}

    vary_dict=deepcopy(components)
    for i in range(len(vary_dict)):
        if (vary_dict[i]['type'] == 'sersic'):
            vary_dict[i]['mu_0'] = True
            vary_dict[i]['r_0'] = True
            vary_dict[i]['n'] = True
        if (vary_dict[i]['type'] == 'exp_disk'):
            vary_dict[i]['mu_0'] = True
            vary_dict[i]['h'] = True
        if (vary_dict[i]['type'] == 'bar'):
            vary_dict[i]['mu_0'] = False
            vary_dict[i]['r'] = False
            vary_dict[i]['h'] = False
        if (vary_dict[i]['type'] == 'ring'):
            vary_dict[i]['mu_0'] = False
            vary_dict[i]['r'] = False
            vary_dict[i]['w'] = False

    return([components, vary_dict])

def fill_params(components, vary_dict=None, include_log_f=False):
    params = Parameters()
    # params.add('mu_0'+str(0), value=components[0]['mu_0'], min=0)
    for i in range(0, len(components)):
        if (vary_dict == None):
            vary=True
            if components[i]['type'] == 'sersic':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=23., vary=vary)
                params.add('r_0'+str(i), value=components[i]['r_0'], min=0.1, max=4., vary=vary)
                params.add('n'+str(i), value=components[i]['n'], min=0.5, max=5., vary=vary)
            if components[i]['type'] == 'exp_disk':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary)
                params.add('h'+str(i), value=components[i]['h'], min=0.1, max=60., vary=vary)
            if components[i]['type'] == 'bar':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary)
                params.add('r'+str(i), value=components[i]['r'], min=components[i]['']*0.8, max=components[i]['r']*1.2, vary=vary)
                params.add('h'+str(i), value=components[i]['h'], min=1e-3, max=1., vary=vary)
            if components[i]['type'] == 'ring':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary)
                params.add('r'+str(i), value=components[i]['r'], min=components[i]['r']*0.8, max=components[i]['r']*1.2, vary=vary)
                params.add('w'+str(i), value=components[i]['w'], min=0.1, max=3., vary=vary)
        else:
            if components[i]['type'] == 'sersic':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=23., vary=vary_dict[i]['mu_0'])
                params.add('r_0'+str(i), value=components[i]['r_0'], min=0.1, max=4., vary=vary_dict[i]['r_0'])
                params.add('n'+str(i), value=components[i]['n'], min=0.5, max=5., vary=vary_dict[i]['n'])
            if components[i]['type'] == 'exp_disk':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary_dict[i]['mu_0'])
                params.add('h'+str(i), value=components[i]['h'], min=0.1, max=60., vary=vary_dict[i]['h'])         
            if components[i]['type'] == 'bar':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary_dict[i]['mu_0'])
                params.add('r'+str(i), value=components[i]['r'], min=components[i]['r']*0.8, max=components[i]['r']*1.2, vary=vary_dict[i]['r'])
                params.add('h'+str(i), value=components[i]['h'], min=1e-3, max=1., vary=vary_dict[i]['h'])
            if components[i]['type'] == 'ring':
                params.add('mu_0'+str(i),   value=components[i]['mu_0'], min=10., max=30., vary=vary_dict[i]['mu_0'])
                params.add('r'+str(i), value=components[i]['r'], min=components[i]['r']*0.8, max=components[i]['r']*1.2, vary=vary_dict[i]['r'])
                params.add('w'+str(i), value=components[i]['w'], min=0.1, max=3., vary=vary_dict[i]['w'])
        if (include_log_f):
            params.add('log_f', value=-3., min=-np.inf, max=np.inf, vary=True)
    return(params)

def gaussian(x, wid, mean=0., amp=1):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-mean)**2 / (2*wid**2))

def make_psf_profile(psf2d_file, show_plot=False):
    psf_data = fits.getdata(psf2d_file)
    center_x, center_y = np.array(psf_data.shape) // 2
    y, x = np.indices(psf_data.shape)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_unique = np.unique(np.round(r).astype(int))
    radial_profile = np.array([psf_data[r == radius].mean() for radius in r_unique])
    gmodel = Model(gaussian)
    r_unique = r_unique[~np.isnan(radial_profile)]
    radial_profile = radial_profile[~np.isnan(radial_profile)]
    resfit = gmodel.fit(radial_profile, x=r_unique, amp=0.5, wid=1)  
    #print(resfit.fit_report())
    xhr = np.linspace(0, r_unique[-1], 1000)
    amp = resfit.params['amp'].value
    wid = resfit.params['wid'].value

    return amp, wid

def extract_psf_photutils(psf2d):
    geometry = EllipseGeometry(x0=psf2d.shape[0]/2.0, y0=psf2d.shape[1]/2.0, sma=5., eps=0.00,
                            pa=10.0 * np.pi / 180.0, fix_pa=True, fix_eps=True)
    ellipse = Ellipse(psf2d, geometry)
    isolist = ellipse.fit_image(linear=True, step=1)
    t_ = isolist.to_table()

    t = t_[t_['sma'].data < 20.0]

    psf_ = t['intens'].data / np.sum(t['intens'].data) / 2.0
    psf_ = psf_ / np.sum(psf_)
    psf = np.concatenate((np.flip(psf_), psf_[1:]))
    psf = psf / np.sum(psf)
    # set_trace()
    return(psf)

def flx2mag(flux, zeropoint, scale):
    return zeropoint - 2.5*np.log10(flux/scale**2.)

def mag2flx(m, zeropoint, scale):
    return scale ** 2. * np.float_power(10., 0.4 * (zeropoint - m))

def calc_incl_corr(x, ell, mag, mag_ell=27.):
    xspace = np.linspace(min(x) + 0.01, max(x) - 0.01, num=10000, endpoint=True)
    mag_interp = interp1d(x, mag, kind='cubic')(xspace)
    ell_interp = interp1d(x, ell, kind='cubic')(xspace)
    # el = el(xspace)
    id_incl_corr = np.nanargmin(np.abs(mag_interp - mag_ell))
    cos_i = 1 - ell_interp[id_incl_corr]
    incl_corr = 2.5 * np.log10(cos_i)
    # plt.plot(x, ell, 'bo', markersize=2)
    # plt.ylim(0, 1.2)
    # plt.axvline(xspace[id_incl_corr], label   =f'e = {el[id_incl_corr]}')
    # plt.legend()
    # if show_plot is True:
    #     plt.show()
    # if save_plot is True:
    #     plt.savefig(f'{dir_path}/{dir_phot_res}/{file_name}_ellipticity_plot.png', dpi=400)
    # plt.clf()
    return (incl_corr, ell_interp[id_incl_corr])

def calc_extinction(ra, dec, R=3.303):
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    csfd = CSFDQuery()
    ebv = csfd(coords)
    er = ebv*R# 2.285
    return er

def calc_dimming(z):
    return 10. * np.log10(z + 1.)

def b(n):
    return 2*n - 1.0/3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)

def make_model(params, components, 
               pixsize, scale_kpc, zeropoint,
               x_kpc, 
               psf,
               n_oversample=10,
               return_hr=False
               ):
    '''
    Creating photometric profile model consisting of Sersic profiles and exponential disks

    Attributes
    ----------

    params : (lmfit.Parameters)
        parameters of the profile
    components : (dict)

    pixsize : (float) arcsec/pixel
        size of one pixel, arcsec per pixel
    scale_kpc : (float) kpc/arcsec
        size of one arcsec in kpc, kpc per arcsec
    zeropoint : (float)
        magnitude zeropoint
    x_kpc : (float 1D array) kpc
        regular 1D grid in kpc
    sigma_psf : (float) arcsec
        sigma of the PSF in arcsec
    n_oversample : (int)
        oversampling factor for x_kpc
    '''
    # usually step is pixsize*scale
    x_prof = np.arange(min(x_kpc)*0.99, 1.5*max(x_kpc), pixsize*scale_kpc)
    
    I_sum = np.zeros(x_prof.shape)
    I_arr = []
    for i in range(len(components)):
        if components[i]['type']=='sersic':
            mu_0 = params['mu_0'+str(i)].value
            I_0 = mag2flx(mu_0, zeropoint=zeropoint, scale=pixsize)
            r0 = params['r_0'+str(i)].value
            n = params['n'+str(i)].value
            p = I_0*np.exp(-b(n)*((x_prof/r0)**(1.0/n) - 1.0))
            I_sum = I_sum + p
            I_arr.append(p)
        if components[i]['type']=='exp_disk':
            mu_0 = params['mu_0'+str(i)].value
            I_0 = mag2flx(mu_0, zeropoint=zeropoint, scale=pixsize)
            h = params['h'+str(i)].value
            p = I_0*np.exp(-x_prof/h)
            I_sum = I_sum + p
            I_arr.append(p)
        if components[i]['type']=='bar':
            mu_0 = params['mu_0'+str(i)].value
            I_0 = mag2flx(mu_0, zeropoint=zeropoint, scale=pixsize)
            r = params['r'+str(i)].value
            h = params['h'+str(i)].value
            p = I_0/(np.exp((x_prof-r)/h)+1.)
            I_sum = I_sum + p
            I_arr.append(p)
        if components[i]['type']=='ring':
            mu_0 = params['mu_0'+str(i)].value
            I_0 = mag2flx(mu_0, zeropoint=zeropoint, scale=pixsize)
            r = params['r'+str(i)].value
            w = params['w'+str(i)].value
            p = gaussian(x_prof, wid=w, mean=r, amp=I_0)
            I_sum = I_sum + p
            I_arr.append(p)            
    idx_left = len(x_prof)-1
    I_left = np.flip(I_sum[:idx_left+1])
    I_to_convolve = np.concatenate((I_left, I_sum))
    I_convolved = convolve_fft(I_to_convolve,psf)
    
    model_I = I_convolved[idx_left+1:]
    model_mag = flx2mag(model_I, zeropoint=zeropoint, scale=pixsize)
    f_mag = interp1d(x_prof, model_mag)
    f_I = interp1d(x_prof, model_I)
    model_mag_pix = f_mag(x_kpc)
    model_I_pix = f_I(x_kpc)
    mu_arr = []
    if (return_hr):
        mu_arr_hr = []
        mu_arr_hr_convolved = []
    for i in range(len(I_arr)):
        mu_mod = flx2mag(I_arr[i], zeropoint=zeropoint, scale=pixsize)
        f = interp1d(x_prof, mu_mod)
        mu_mod_pix = f(x_kpc)
        mu_arr.append(mu_mod_pix)
        if (return_hr):
            comp_to_convolve = np.concatenate((np.flip(I_arr[i][:idx_left+1]), I_arr[i]))
            comp_convolved = convolve_fft(comp_to_convolve,psf)
            mu_mod_convolved = flx2mag(comp_convolved[idx_left+1:], zeropoint=zeropoint, scale=pixsize)
            mu_arr_hr.append(mu_mod)
            mu_arr_hr_convolved.append(mu_mod_convolved)
    if (return_hr):
        return [x_prof, model_mag, mu_arr_hr, mu_arr_hr_convolved, model_mag_pix, x_kpc]
    else:
        return [model_I_pix, model_mag_pix, mu_arr]

def log_likelihood(params, components, x_kpc, data, data_err, pixsize, scale_kpc, zeropoint, psf):

    model_pix = make_model(params, components,
                           pixsize=pixsize, scale_kpc=scale_kpc, zeropoint=zeropoint,
                           x_kpc=x_kpc, 
                           psf=psf)[0]
    s2 = data_err**2+model_pix**2*np.exp(2.*params['log_f'].value)
    return -0.5*np.nansum((model_pix-data)**2/s2 + np.log(2.*np.pi*s2))

def fcn2min(params, components, x_kpc, data, data_err, pixsize, scale_kpc, zeropoint, psf):
    '''
    Outputs error-normalised residuals with respect to the model to be minimised

    Attributes
    ----------

    params, components, pixsize, scale_kpc, zeropoint, x_kpc, sigma_psf :
        see make_model()

    data : (float 1D array)
        array with data values
    data_err : (float 1D array)
        array with error values

    '''
    model_pix = make_model(params, components, 
                           pixsize=pixsize, scale_kpc=scale_kpc, zeropoint=zeropoint,
                           x_kpc=x_kpc, 
                           psf=psf)[0]
    return (model_pix - data) / data_err

def fcn2min_emcee(params, components, x_kpc, data, data_err, pixsize, scale_kpc, zeropoint, psf):
    '''
    Outputs error-normalised residuals with respect to the model to be minimised

    Attributes
    ----------

    params, components, pixsize, scale_kpc, zeropoint, x_kpc, sigma_psf :
        see make_model()

    data : (float 1D array)
        array with data values
    data_err : (float 1D array)
        array with error values

    '''
    model_pix = make_model(params, components, 
                           pixsize=pixsize, scale_kpc=scale_kpc, zeropoint=zeropoint,
                           x_kpc=x_kpc, 
                           psf=psf)[0]
    return -0.5*np.nansum((model_pix - data)**2./data_err**2. + np.log(2.*np.pi*data_err**2.))

def fcn2min_ML(params, components, x_kpc, data, data_err, pixsize, scale_kpc, zeropoint, psf):
    return -log_likelihood(params, components, x_kpc, data, data_err, pixsize, scale_kpc, zeropoint, psf)

def fill_tbl_fit_params(tbl, params, comp_name, idx_comp, param_names):
    for name in param_names:
        val_arr, err_arr, vary_arr, bounds_arr = [], [], [], []
        for i in idx_comp:
            val_arr.append(params[f"{name}{i}"].value)
            err_arr.append(params[f"{name}{i}"].stderr)
            vary_arr.append(params[f"{name}{i}"].vary)
            bounds_arr.append([params[f"{name}{i}"].min,params[f"{name}{i}"].max])
        val_arr = np.array(val_arr).astype(float)
        err_arr = np.array(err_arr).astype(float)
        vary_arr = np.array(vary_arr).astype(int)
        bounds_arr = np.array(bounds_arr).astype(float)
        tbl.add_column([val_arr], name=f"{comp_name}_{name}")
        tbl.add_column([err_arr], name=f"{comp_name}_{name}_err")
        tbl.add_column([vary_arr], name=f"{comp_name}_{name}_vary")
        tbl.add_column([bounds_arr], name=f"{comp_name}_{name}_bounds")


def export_result(objname,
                  intens, intens_err, pixmask, psf, model_hr, components, params, chisqr, dof, 
                  zeropoint, pixsize, scale_kpc,
                  z, r_hl,
                  H, omega_m, ell,
                  bic_lmfit,
                  fout,
                  bic=None):
    component_types = []
    for i in range(len(components)):
        component_types.append(components[i]['type'])
    component_types = np.array(component_types)
    idx_sersic = (component_types == 'sersic').nonzero()[0]
    idx_exp_disk = (component_types == 'exp_disk').nonzero()[0]
    idx_bar = (component_types == 'bar').nonzero()[0]
    idx_ring = (component_types == 'ring').nonzero()[0]
    tbl = Table([[objname]], names=['name'])
    tbl.add_column([H], name='H')
    tbl.add_column([omega_m], name='omega_m')
    tbl.add_column([psf], name='psf')
    tbl.add_column([pixsize], name='pixsize')
    tbl.add_column([z], name='z')
    tbl.add_column([scale_kpc], name='scale_kpc')
    tbl.add_column([r_hl], name='re_prof')
    tbl.add_column([model_hr[5]], name='r_prof')
    tbl.add_column([intens], name='intens_prof')
    tbl.add_column([intens_err], name='intens_prof_err')
    tbl.add_column([flx2mag(intens, zeropoint, pixsize)], name='mag_prof')
    tbl.add_column([flx2mag(intens+intens_err, zeropoint, pixsize)], name='mag_prof_err_plus')
    tbl.add_column([flx2mag(intens-intens_err, zeropoint, pixsize)], name='mag_prof_err_minus')
    tbl.add_column([model_hr[4]], name='mag_model_pix')
    tbl.add_column([model_hr[0]], name='r_model')
    tbl.add_column([pixmask], name='pixmask')
    tbl.add_column([model_hr[1]], name='mag_model')
    tbl.add_column([model_hr[2]], name='mag_comps')
    tbl.add_column([model_hr[3]], name='mag_comps_conv')
    tbl.add_column([component_types], name='comps_names')
    if (len(idx_sersic) != 0):
        fill_tbl_fit_params(tbl=tbl, params=params, comp_name='sersic', idx_comp=idx_sersic, param_names=["mu_0", "r_0", "n"])
    if (len(idx_exp_disk) != 0):
        fill_tbl_fit_params(tbl=tbl, params=params, comp_name='exp_disk', idx_comp=idx_exp_disk, param_names=["mu_0", "h"])
    if (len(idx_bar) != 0):
        fill_tbl_fit_params(tbl=tbl, params=params, comp_name='bar', idx_comp=idx_bar, param_names=["mu_0", "r", "h"])
    if (len(idx_ring) != 0):
        fill_tbl_fit_params(tbl=tbl, params=params, comp_name='ring', idx_comp=idx_ring, param_names=["mu_0", "r", "w"])
    tbl.add_column(chisqr, name='chi2')
    tbl.add_column(dof, name='dof')
    tbl.add_column(chisqr/dof, name='chi2/dof')
    tbl.add_column(ell, name='ellipticity')
    tbl.add_column(bic_lmfit, name='bic_lmfit')
    if (bic != None):
        tbl.add_column(bic, name='bic')

    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(tbl)])
    hdul.writeto(fout, overwrite=True)

def mcmc_fit_wrapper(args_dict):
    print("running wrapper")
    run_fit_mcmc(**args_dict)

def ML_fit_wrapper(args_dict):
    print("running wrapper")
    run_fit_ML(**args_dict)


def fit_sample_multithread(objnames, param_arr_all,
                           pixscale, zeropoint,
                           iso_table_dir, psf_dir, auto_fit_res_dir, cutout_dir,
                           dirout, plot_dir,
                           iso_table_suf='.tab', psf_suf='-psf1d.fits', auto_fit_res_suf='_decomp.fits', cutout_suf='.jpg',
                           correct_inclination=True, correct_dimming=True, correct_extinction=True,
                           n_threads=1,
                           fit_1disk=False,
                           minimzation='emcce',
                           param_colnames={"objname":"objname",
                                           "ra":"ra",
                                           "dec":"dec",
                                           "z":"z"}):
    
    args_dict_list = []
    for objname in objnames:
        iso_table = Table.read(f"{iso_table_dir}/{objname}{iso_table_suf}", format='ascii')
        psf1d = fits.open(f"{psf_dir}/{objname}{psf_suf}")[1].data
        auto_fit_res_file = f"{auto_fit_res_dir}/{objname}{auto_fit_res_suf}"
        auto_fit_res = fits.open(auto_fit_res_file)[1].data

        pixmask = auto_fit_res["pixmask"][0]
        r_hl = auto_fit_res["re_prof"][0]
        components, vary_dict = fill_components(auto_fit_res_file)
        if fit_1disk:
            if (len(components)>2):
                if (components[2]["type"]=='exp_disk'):
                    components_1disk={0: components[0], 1: components[2]}
                    vary_dict_1disk ={0: vary_dict[0], 1: vary_dict[2]}
                    if (len(components)>3):
                        for i in range(3, len(components)):
                            components_1disk[i-1] = components[i]
                            vary_dict_1disk[i-1] = vary_dict[i]
                    components = components_1disk
                    vary_dict = vary_dict_1disk
                else:
                    continue
            else:
                continue

        obj_idx = (param_arr_all[param_colnames["objname"]] == objname).nonzero()[0]
        ra = param_arr_all[param_colnames["ra"]][obj_idx]
        dec = param_arr_all[param_colnames["dec"]][obj_idx]
        z = param_arr_all[param_colnames["z"]][obj_idx]

        args_dict = {
            'iso_table':iso_table, 
            'ra':ra, 
            'dec':dec, 
            'z':z, 
            'r_hl':r_hl, 
            'components':components, 'vary_dict':vary_dict, 'pixmask':pixmask,
            'pixsize':pixscale, 'zeropoint':zeropoint, 
            'psf2d':None, 'psf1d':psf1d, 'sigma_psf':None, # sigma_psf is assumed to be in arcsec
            'dirout':dirout, 'outfname':f"{objname}_decomp_mcmc.fits", 'objname':objname, 'plot_dir':plot_dir,
            'correct_inclination':correct_inclination, 'correct_dimming':correct_dimming, 'correct_extinction':correct_extinction,
            'constr_arr':None,
            'plot':True, 'cutout_file':f"{cutout_dir}/{objname}{cutout_suf}"
        }

        args_dict_list.append(args_dict)

    with Pool(processes=n_threads) as pool:
        match minimzation:
            case 'emcee':
                result = pool.map(mcmc_fit_wrapper, args_dict_list)
            case 'ML':
                result = pool.map(ML_fit_wrapper, args_dict_list)

#        for result in result.get():
#            print(f'Got result: {result}', flush=True)

def run_fit_ML(iso_table, ra, dec, z, r_hl, components, vary_dict, pixmask,
                   pixsize, zeropoint, 
                   psf2d=None, psf1d=None, sigma_psf=None, # sigma_psf is assumed to be in arcsec
                   dirout='./', outfname='test', objname='test', plot_dir='./',
                   correct_inclination=True, correct_dimming=True, correct_extinction=True,
                   constr_arr=None,
                   include_log_f=True,
                   plot=False, cutout_file='./'):
    print("running")
    H = 67.4
    omega_m = 0.315
    cosmo = FlatLambdaCDM(H0=H, Om0=omega_m) 
    scale_kpc = cosmo.angular_diameter_distance(z).value*np.pi/180./3600.*1000.

    if (psf2d is not None):
        psf = extract_psf_photutils(psf2d)
    if (psf1d is not None):
        psf = psf1d
    if (sigma_psf is not None):
        x_psf = np.arange(start=-20., stop=20., step=1.)
        psf = gaussian(x_psf, sigma_psf/pixsize)
    r_kpc = iso_table['sma'].data*pixsize*scale_kpc
    mag = flx2mag(iso_table['intens'].data, zeropoint=zeropoint, scale=pixsize)
    rel_err = iso_table['intens_err'].data/iso_table['intens'].data*2.5*np.log10(np.e)
    ell = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[1]
    
    correction = 0.
    if (correct_extinction):
        extinction_corr = calc_extinction(ra, dec)
        correction -= extinction_corr
    if (correct_dimming):
        dimming_corr = calc_dimming(float(z))
        correction -= dimming_corr
    if (correct_inclination):
        inclination_corr = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[0]
        correction -= inclination_corr

    mag += correction
    intens = mag2flx(mag, zeropoint=zeropoint, scale=pixsize)
    intens_err = iso_table['intens_err'].data*1.#np.sqrt(iso_table["ndata"].data)

    r_kpc_in = r_kpc*1.
    mag_in = mag*1.
    intens_in = intens*1.
    intens_err_in = intens_err*1.
    bdata_idx = (pixmask == 0).nonzero()[0]
    intens[bdata_idx] = np.nan

    params = fill_params(components, vary_dict, log_f=True)
    if (constr_arr != None):
        for constr in constr_arr:
            params[f"{constr['param']}{constr['comp']}"].min = constr["win"][0]
            params[f"{constr['param']}{constr['comp']}"].max = constr["win"][1]

    print(f"Fitting {objname}")
    minner = Minimizer(fcn2min_ML, params, fcn_args=(components, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
        nan_policy='omit')
    
    result = minner.minimize(method='SLSQP')

    # model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf)
    model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf, return_hr=True)
    export_result(objname, 
                  intens_in, intens_err_in, (~np.isnan(intens)).astype(int), psf, model, components, result.params, result.chisqr, result.nfree,
                  zeropoint, pixsize, scale_kpc, 
                  z, r_hl, 
                  H, omega_m, ell, 
                  result.bic,
                  f"{dirout}/{outfname}.fits")

    if (plot):
        plot_decomposition(f"{dirout}/{outfname}.fits", cutout_file, f"{plot_dir}{objname}.pdf")
        emcee_plot = corner.corner(result.flatchain, labels=result.var_names,
                               bins=10)
        emcee_plot.savefig(f"{plot_dir}/cornerplot_{objname}.pdf")
    print(f"{objname} fitted")
    return(result)

def run_fit_mcmc(iso_table, ra, dec, z, r_hl, components, vary_dict, pixmask,
                   pixsize, zeropoint, 
                   psf2d=None, psf1d=None, sigma_psf=None, # sigma_psf is assumed to be in arcsec
                   dirout='./', outfname='test', objname='test', plot_dir='./',
                   correct_inclination=True, correct_dimming=True, correct_extinction=True,
                   constr_arr=None,
                   plot=False, cutout_file='./'):
    print("running")
    H = 67.4
    omega_m = 0.315
    cosmo = FlatLambdaCDM(H0=H, Om0=omega_m) 
    scale_kpc = cosmo.angular_diameter_distance(z).value*np.pi/180./3600.*1000.

    if (psf2d is not None):
        psf = extract_psf_photutils(psf2d)
    if (psf1d is not None):
        psf = psf1d
    if (sigma_psf is not None):
        x_psf = np.arange(start=-20., stop=20., step=1.)
        psf = gaussian(x_psf, sigma_psf/pixsize)
    r_kpc = iso_table['sma'].data*pixsize*scale_kpc
    mag = flx2mag(iso_table['intens'].data, zeropoint=zeropoint, scale=pixsize)
    rel_err = iso_table['intens_err'].data/iso_table['intens'].data*2.5*np.log10(np.e)
    ell = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[1]
    
    correction = 0.
    if (correct_extinction):
        extinction_corr = calc_extinction(ra, dec)
        correction -= extinction_corr
    if (correct_dimming):
        dimming_corr = calc_dimming(float(z))
        correction -= dimming_corr
    if (correct_inclination):
        inclination_corr = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[0]
        correction -= inclination_corr

    mag += correction
    intens = mag2flx(mag, zeropoint=zeropoint, scale=pixsize)
    intens_err = iso_table['intens_err'].data*1.#np.sqrt(iso_table["ndata"].data)

    r_kpc_in = r_kpc*1.
    mag_in = mag*1.
    intens_in = intens*1.
    intens_err_in = intens_err*1.
    bdata_idx = (pixmask == 0).nonzero()[0]
    intens[bdata_idx] = np.nan

    params = fill_params(components, vary_dict)
    if (constr_arr != None):
        for constr in constr_arr:
            params[f"{constr['param']}{constr['comp']}"].min = constr["win"][0]
            params[f"{constr['param']}{constr['comp']}"].max = constr["win"][1]

    print(f"Fitting {objname}")
    minner = Minimizer(fcn2min_emcee, params, fcn_args=(components, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
        nan_policy='omit',burn=500, steps=5000, thin=25,
                      float_behavior='posterior', progress=True)
    result = minner.minimize(method='emcee')
    result.flatchain.to_csv(f"{dirout}/{objname}_flatchain.csv")

    #calculating BIC
    log_L = fcn2min_emcee(result.params, components, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf)
    bic = result.nvarys*np.log(result.ndata) - 2*log_L

    model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf)
    model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf, return_hr=True)
    export_result(objname, 
                  intens_in, intens_err_in, (~np.isnan(intens)).astype(int), psf, model, components, result.params, result.chisqr, result.nfree,
                  zeropoint, pixsize, scale_kpc, 
                  z, r_hl, 
                  H, omega_m, ell, 
                  result.bic,
                  f"{dirout}/{outfname}.fits")
    result.flatchain.to_csv(f"{dirout}/{objname}_flatchain.csv")

    if (plot):
        plot_decomposition(f"{dirout}/{outfname}.fits", cutout_file, f"{plot_dir}{objname}.pdf")
        emcee_plot = corner.corner(result.flatchain, labels=result.var_names,
                               bins=10)
        emcee_plot.savefig(f"{plot_dir}/cornerplot_{objname}.pdf")
    print(f"{objname} fitted")
    return(result)

def run_fit_manual(iso_table, ra, dec, z, r_hl, components, vary_dict, pixmask,
                   pixsize, zeropoint, 
                   psf2d=None, psf1d=None, sigma_psf=None, # sigma_psf is assumed to be in arcsec
                   dirout='./', outfname='test', objname='test', plot_dir='./',
                   correct_inclination=True, correct_dimming=True, correct_extinction=True,
                   constr_arr=None,
                   plot=False, cutout_file='./'):
    
    H = 67.4
    omega_m = 0.315
    cosmo = FlatLambdaCDM(H0=H, Om0=omega_m) 
    scale_kpc = cosmo.angular_diameter_distance(z).value*np.pi/180./3600.*1000.

    if (psf2d is not None):
        psf = extract_psf_photutils(psf2d)
    if (psf1d is not None):
        psf = psf1d
    if (sigma_psf is not None):
        x_psf = np.arange(start=-20., stop=20., step=1.)
        psf = gaussian(x_psf, sigma_psf/pixsize)
    r_kpc = iso_table['sma'].data*pixsize*scale_kpc
    mag = flx2mag(iso_table['intens'].data, zeropoint=zeropoint, scale=pixsize)
    rel_err = iso_table['intens_err'].data/iso_table['intens'].data*2.5*np.log10(np.e)
    ell = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[1]

    correction = 0.
    if (correct_extinction):
        extinction_corr = calc_extinction(ra, dec)
        correction -= extinction_corr
    if (correct_dimming):
        dimming_corr = calc_dimming(float(z))
        correction -= dimming_corr
    if (correct_inclination):
        inclination_corr = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[0]
        correction -= inclination_corr

    mag += correction
    intens = mag2flx(mag, zeropoint=zeropoint, scale=pixsize)
    intens_err = iso_table['intens_err'].data*1.#np.sqrt(iso_table["ndata"].data)

    r_kpc_in = r_kpc*1.
    mag_in = mag*1.
    intens_in = intens*1.
    intens_err_in = intens_err*1.
    bdata_idx = (pixmask == 0).nonzero()[0]
    intens[bdata_idx] = np.nan

    params = fill_params(components, vary_dict)
    if (constr_arr != None):
        for constr in constr_arr:
            params[f"{constr['param']}{constr['comp']}"].min = constr["win"][0]
            params[f"{constr['param']}{constr['comp']}"].max = constr["win"][1]

    minner = Minimizer(fcn2min, params, fcn_args=(components, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
        nan_policy='omit')
    result = minner.minimize(method='leastsq',
                             ftol=1e-1,
                             epsfcn=1e-2)
    model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf)
    model = make_model(result.params, components, pixsize, scale_kpc, zeropoint, r_kpc, psf, return_hr=True)
    export_result(objname, 
                  intens_in, intens_err_in, (~np.isnan(intens)).astype(int), psf, model, components, result.params, result.chisqr, result.nfree,
                  zeropoint, pixsize, scale_kpc, 
                  z, r_hl, 
                  H, omega_m, ell,
                  result.bic,
                  f"{dirout}/{outfname}.fits")
    if (plot):
        plot_decomposition(f"{dirout}/{outfname}.fits", cutout_file, f"{plot_dir}{objname}.pdf")
    return(result)



def run_fit(iso_table, ra, dec, z,
                  pixsize, zeropoint, 
                  psf2d=None, psf1d=None, sigma_psf=None, # sigma_psf is assumed to be in arcsec
                  dirout='./', outfname='test', objname='test', plot_dir='./',
                  correct_inclination=True, correct_dimming=True, correct_extinction=True,
                  plot=False, cutout_file='./'):
    
    H = 67.4
    omega_m = 0.315
    cosmo = FlatLambdaCDM(H0=H, Om0=omega_m) 
    scale_kpc = cosmo.angular_diameter_distance(z).value*np.pi/180./3600.*1000.

    if (psf2d is not None):
        psf = extract_psf_photutils(psf2d)
    if (psf1d is not None):
        psf = psf1d
    if (sigma_psf is not None):
        x_psf = np.arange(start=-20., stop=20., step=1.)
        psf = gaussian(x_psf, sigma_psf/pixsize)
    r_kpc = iso_table['sma'].data*pixsize*scale_kpc
    mag = flx2mag(iso_table['intens'].data, zeropoint=zeropoint, scale=pixsize)
    rel_err = iso_table['intens_err'].data/iso_table['intens'].data*2.5*np.log10(np.e)
    ell = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[1]

    correction = 0.
    if (correct_extinction):
        extinction_corr = calc_extinction(ra, dec)
        correction -= extinction_corr
    if (correct_dimming):
        dimming_corr = calc_dimming(float(z))
        correction -= dimming_corr
    if (correct_inclination):
        inclination_corr = calc_incl_corr(iso_table['sma'].data, iso_table['ellipticity'].data, mag, mag_ell=27.)[0]
        correction -= inclination_corr

    mag += correction
    intens = mag2flx(mag, zeropoint=zeropoint, scale=pixsize)
    intens_err = iso_table['intens_err'].data*1.#np.sqrt(iso_table["ndata"].data)

    r_kpc_in = r_kpc*1.
    mag_in = mag*1.
    intens_in = intens*1.
    intens_err_in = intens_err*1.

    psf_midx = find_peaks(psf)[0]
    fwhm = peak_widths(psf, psf_midx, 0.5)[0][0]
    psf_hwhm_idx = (r_kpc < fwhm*pixsize*scale_kpc/2.).nonzero()[0]
    if (len(psf_hwhm_idx) != 0):
        fit_idx_min = 2#np.max(psf_hwhm_idx)
    else:
        fit_idx_min = 0

    #Step 0 : check if surfave brightness is constant on the end of the profile
    idx = np.arange(len(r_kpc)*9//10, len(r_kpc), 1)
    p = np.polyfit(r_kpc[idx], mag[idx], deg=2)
    # p_prime = np.array([3.*p[0], 2.*p[1], p[2]])
    p_prime = np.array([2.*p[0], p[1]])
    if((p[0]<0.) & (len(idx)>3)):
        prof_prime = np.polyval(p_prime, r_kpc)
        prof_bidx = (prof_prime < 0.001).nonzero()[0]
        if (len(prof_bidx) != 0):
            fit_idx_max = np.min(prof_bidx)
        else:
            fit_idx_max = len(r_kpc)-1
    else:
        fit_idx_max = len(r_kpc)-1

    r_kpc = r_kpc
    mag[0:fit_idx_min] = np.nan
    mag[fit_idx_max:] = np.nan
    intens[0:fit_idx_min] = np.nan
    intens[fit_idx_max:] = np.nan
    intens_err[0:fit_idx_min] = np.nan 
    intens_err[fit_idx_max:] = np.nan

    # bbidx = ((r_kpc>10.) & (r_kpc<20.)).nonzero()[0]
    # intens[bbidx] = np.nan

    #Step 1 : determine half-light radius of the profile
    flux = mag2flx(mag_in, zeropoint=zeropoint, scale=pixsize)*iso_table["ndata"]*pixsize**2
    curve_of_growth = np.zeros(len(flux))
    for i in range(len(flux)):
        curve_of_growth[i]=np.sum(flux[0:i+1])
    curve_of_growth = curve_of_growth/np.sum(flux)
    halflight_idx = np.max((curve_of_growth < 0.5).nonzero()[0])+1
    r_hl = r_kpc_in[halflight_idx]
    if (r_hl > 10.):
        r_hl = 4.

    inner_disk_idx = ((r_kpc > 10.*r_hl) & (mag < 27.)).nonzero()[0]
    if (len(inner_disk_idx)==0):
        inner_disk_idx = ((r_kpc > 3.*r_hl) & (mag < 27.)).nonzero()[0]
    inner_idx = (mag).nonzero()[0]
    slope, intercept = np.polyfit(r_kpc[inner_disk_idx], mag[inner_disk_idx], deg=1)
    components_tmp = {0:{'type':'sersic', 'mu_0':mag[halflight_idx], 'r_0':r_hl, 'n':3.},
                      1:{'type':'exp_disk', 'mu_0':intercept, 'h':2.5*np.log10(np.e)/slope},
                      }
    vary_dict      = {0:{'type':'sersic', 'mu_0':True, 'r_0':True, 'n':True},
                      1:{'type':'exp_disk', 'mu_0':True, 'h':True}}
    params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)
    minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc[inner_idx], intens[inner_idx], intens_err[inner_idx], pixsize, scale_kpc, zeropoint, psf),
        nan_policy='omit')
    try:
        result = minner_tmp.minimize(method='leastsq')
        r0_guess = result.params['r_00'].value
        mu0disk_guess = result.params["mu_01"].value
        h_guess = result.params["h1"].value
    except ValueError:
        model_ini = make_model(params_tmp, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf)
        if (mag[0]-model_ini[0][0] >= 1.5):
            components_tmp[0]["mu_0"] = mag[halflight_idx]+1.
            params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)
            minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc[inner_idx], intens[inner_idx], intens_err[inner_idx], pixsize, scale_kpc, zeropoint, psf),
                nan_policy='omit')
            try:
                result = minner_tmp.minimize(method='leastsq')
            except ValueError:
                components_tmp[0]["mu_0"] = mag[halflight_idx]+2.
                params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)
                minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc[inner_idx], intens[inner_idx], intens_err[inner_idx], pixsize, scale_kpc, zeropoint, psf),
                    nan_policy='omit')
                result = minner_tmp.minimize(method='leastsq')
            r0_guess = result.params['r_00'].value
            mu0disk_guess = result.params["mu_01"].value
            h_guess = result.params["h1"].value
        else:
            params_tmp['r_00'].max = 3.*r_hl
            params_tmp['n0'].max = 8.
            minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc[inner_idx], intens[inner_idx], intens_err[inner_idx], pixsize, scale_kpc, zeropoint, psf),
                            nan_policy='omit')
            result = minner_tmp.minimize(method='leastsq')
            r0_guess = r_hl
            mu0disk_guess = intercept
            h_guess = 2.5*np.log10(np.e)/slope

    #Searching for bars and rings
    model = make_model(result.params, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf)
    resid =  model[1]-mag
    idx_barreg = ((r_kpc > 3.*r_hl) & (mag < 26.5)).nonzero()[0]
    if (len(idx_barreg)!=0):
        if (len(resid[idx_barreg]<0.5) > len(idx_barreg)*0.7):
            mu0disk_guess = intercept
            h_guess = 2.5*np.log10(np.e)/slope
            result.params["mu_01"].value = intercept
            result.params["h1"].value = 2.5*np.log10(np.e)/slope
            model = make_model(result.params, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf)
            resid =  model[1]-mag

        resid_barreg = resid[idx_barreg]
        peak_idxs = find_peaks(resid_barreg, height=0.15, distance=3)[0]+idx_barreg[0]
        # peak_idxs = peak_local_max(resid_barreg, threshold_abs=0.15, min_distance=3)+idx_barreg[0]
        # peak_idxs = argrelextrema(resid_barreg, np.greater)[0]+idx_barreg[0]
        # gheight_idx = (resid[peak_idxs] > np.median(resid_barreg)+0.15).nonzero()[0]
        # peak_idxs = peak_idxs[gheight_idx]
        fit_bar_rings=(len(peak_idxs) !=0)
        if (fit_bar_rings):
            if (r_kpc[peak_idxs[0]] < np.nanmax([5.*r_hl, 2.*h_guess])):
                bar_r = r_kpc[peak_idxs[0]]
                bar_mag = mag[peak_idxs[0]]
                ring_radii = r_kpc[peak_idxs[1:]]
                ring_mags = mag[peak_idxs[1:]]
                bar=True
                n_rings = len(peak_idxs)-1
            else:
                ring_radii = r_kpc[peak_idxs]
                ring_mags = mag[peak_idxs]
                n_rings = len(peak_idxs)
                bar=False
            components_tmp = {0:{'type':'sersic', 'mu_0':result.params["mu_00"].value, 'r_0':r0_guess, 'n':result.params["n0"].value},
                              1:{'type':'exp_disk', 'mu_0':mu0disk_guess, 'h':h_guess}}
            vary_dict      = {0:{'type':'sersic', 'mu_0':True, 'r_0':True, 'n':True},
                              1:{'type':'exp_disk', 'mu_0':True, 'h':False}}
            if (bar):
                components_tmp[2] = {'type':'bar', 'mu_0':bar_mag, 'r':bar_r, 'h':bar_r/100.}
                vary_dict[2] = {'type':'bar', 'mu_0':True, 'r':True, 'h':True}
                i0=3
            else:
                i0=2
            for i in range(n_rings):
                components_tmp[i0+i] = {'type':'ring', 'mu_0':ring_mags[i], 'r':ring_radii[i], 'w':1.}
                vary_dict[i0+i] = {'type':'ring', 'mu_0':True, 'r':True, 'w':True}
            params_tmp = fill_params(components_tmp, vary_dict=vary_dict)
            # for i in range(len(peak_idxs)):
            #     params_tmp[f"r_bar{2+i}"].min = r_kpc[min(idx_barreg)]
            #     params_tmp[f"r_bar{2+i}"].max = r_kpc[max(idx_barreg)]
            minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc[inner_idx], intens[inner_idx], intens_err[inner_idx], pixsize, scale_kpc, zeropoint, psf),
                            nan_policy='omit')
            result = minner_tmp.minimize(method='leastsq')
    else:
        fit_bar_rings=False

    result_inter = result

    for i in range(len(components_tmp)):
        bad_ring_idx = []
        if (components_tmp[i]["type"] == "ring"):
            r_ring = result.params[f"r{i}"]
            w_ring = result.params[f"w{i}"]
            pix_fwhm_reg_idx = ((r_kpc > r_ring-2.355*w_ring/2.) & (r_kpc < r_ring+2.355*w_ring/2.)).nonzero()[0]
            if (len(pix_fwhm_reg_idx) <= 3):
                bad_ring_idx.append(i)

    #Searching for the second disk
    model = make_model(result.params, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf)
    resid =  model[1]-mag
    bresid_idx = (resid > 0.2).nonzero()[0]
    r_disk2 = r_kpc[bresid_idx]
    disk2_r = np.median(r_disk2)
    fit_2_disks = (len(bresid_idx) >= 3)#((disk2_r > result.params["h1"]) and (len(bresid_idx) >= 3))
    disk2_idx = ((mag > 26.) & (~np.isnan(mag))).nonzero()[0]
    slope2, intercept2 = np.polyfit(r_kpc[disk2_idx], mag[disk2_idx], deg=1)
    h2 = 2.5*np.log10(np.e)/slope2
    mu0disk_guess = intercept
    h_guess = 2.5*np.log10(np.e)/slope
    if (intercept2 < np.nanmin(mag)):
        fit_2_disks=False
    if (fit_2_disks):
        components_tmp = {0:{'type':'sersic', 'mu_0':result.params["mu_00"].value, 'r_0':r0_guess, 'n':result.params["n0"].value},
                          1:{'type':'exp_disk', 'mu_0':mu0disk_guess, 'h':h_guess},
                          2:{'type':'exp_disk', 'mu_0':intercept2, 'h':2.5*np.log10(np.e)/slope2},
                          }
        vary_dict      = {0:{'type':'sersic', 'mu_0':True, 'r_0':True, 'n':False},
                          1:{'type':'exp_disk', 'mu_0':True, 'h':True},
                          2:{'type':'exp_disk', 'mu_0':False, 'h':False}}
        i0=3
    else:
        components_tmp = {0:{'type':'sersic', 'mu_0':result.params["mu_00"].value, 'r_0':r0_guess, 'n':result.params["n0"].value},
                          1:{'type':'exp_disk', 'mu_0':mu0disk_guess, 'h':h_guess}}
        vary_dict      = {0:{'type':'sersic', 'mu_0':True, 'r_0':True, 'n':False},
                          1:{'type':'exp_disk', 'mu_0':True, 'h':True}}
        i0=2

    if (fit_bar_rings):
        if (bar):
            components_tmp[i0] = {'type':'bar', 'mu_0':result.params[f"mu_0{2}"].value, 'r':result.params[f"r{2}"].value, 'h':result.params[f"h{2}"].value}
            vary_dict[i0]      = {'type':'bar', 'mu_0':True, 'r':False, 'h':False}
            idx_ring_cur = i0+1
            for i in range(n_rings):
                idx_ring_result = 3+i
                if (idx_ring_result in bad_ring_idx):
                    continue
                components_tmp[idx_ring_cur] = {'type':'ring', 'mu_0':result.params[f"mu_0{idx_ring_result}"].value, 'r':result.params[f"r{idx_ring_result}"].value, 'w':result.params[f"w{idx_ring_result}"].value}
                vary_dict[idx_ring_cur]      = {'type':'ring', 'mu_0':True, 'r':False, 'w':False}
                idx_ring_cur+=1
        else:
            idx_ring_cur = i0
            for i in range(n_rings):
                idx_ring_result = 2+i
                if (idx_ring_result in bad_ring_idx):
                    continue
                components_tmp[i0+i] = {'type':'ring', 'mu_0':result.params[f"mu_0{idx_ring_result}"].value, 'r':result.params[f"r{idx_ring_result}"].value, 'w':result.params[f"w{idx_ring_result}"].value}
                vary_dict[i0+i]      = {'type':'ring', 'mu_0':True, 'r':False, 'w':False}           
        
    
    params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)   
    params_tmp["h1"].min = 3.*r_hl
    if (fit_2_disks):
        params_tmp["h2"].min = 2.5*np.log10(np.e)/slope2*0.5
        params_tmp["h2"].max = 2.5*np.log10(np.e)/slope2*1.5
        params_tmp["mu_02"].min = intercept2-0.5
        params_tmp["mu_02"].max = intercept2+0.5
    minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
                    nan_policy='omit')
    result = minner_tmp.minimize(method='leastsq') 

    if(fit_bar_rings):
        n_rings -= len(bad_ring_idx)

    if (fit_2_disks):
        model = make_model(result.params, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf)
        resid = model[1]-mag
        remove_2_disk = ((np.nanmedian(resid[r_kpc > 3.*r_hl]) < -0.2) | (result.params["mu_01"].value>27.) | (result.params["mu_01"].value>intercept2) | (abs(result.params["h1"]-result.params["h2"])<3.))
        if (remove_2_disk):
            if ((np.nanmedian(resid[r_kpc > 3.*r_hl]) < -0.2)):
                adj_mu0disk_guess=1.
            else:
                adj_mu0disk_guess=0.
            result=result_inter
            components_tmp = {0:{'type':'sersic', 'mu_0':result.params["mu_00"].value, 'r_0':r0_guess, 'n':result.params["n0"].value},
                          1:{'type':'exp_disk', 'mu_0':mu0disk_guess+adj_mu0disk_guess, 'h':h_guess}}
            vary_dict      = {0:{'type':'sersic', 'mu_0':True, 'r_0':True, 'n':False},
                              1:{'type':'exp_disk', 'mu_0':True, 'h':True}}
            i0=2
            if (fit_bar_rings):
                if (bar):
                    components_tmp[i0] = {'type':'bar', 'mu_0':result.params[f"mu_0{2}"].value, 'r':result.params[f"r{2}"].value, 'h':result.params[f"h{2}"].value}
                    vary_dict[i0]      = {'type':'bar', 'mu_0':True, 'r':False, 'h':False}
                    for i in range(n_rings):
                        components_tmp[i0+1+i] = {'type':'ring', 'mu_0':result.params[f"mu_0{3+i}"].value, 'r':result.params[f"r{3+i}"].value, 'w':result.params[f"w{3+i}"].value}
                        vary_dict[i0+1+i]      = {'type':'ring', 'mu_0':True, 'r':False, 'w':False}
                else:
                     for i in range(n_rings):
                        components_tmp[i0+i] = {'type':'ring', 'mu_0':result.params[f"mu_0{2+i}"].value, 'r':result.params[f"r{2+i}"].value, 'w':result.params[f"w{2+i}"].value}
                        vary_dict[i0+i]      = {'type':'ring', 'mu_0':True, 'r':False, 'w':False}

            params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)   
            params_tmp["h1"].min = 3.*r_hl
            minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
                        nan_policy='omit')
            result = minner_tmp.minimize(method='leastsq')
            fit_2_disks=False
        else:
            components_tmp = {0:{'type':'sersic', 'mu_0':result.params["mu_00"].value, 'r_0':result.params["r_00"].value, 'n':result.params["n0"].value},
                              1:{'type':'exp_disk', 'mu_0':result.params["mu_01"].value, 'h':result.params["h1"].value},
                              2:{'type':'exp_disk', 'mu_0':result.params["mu_02"].value, 'h':result.params["h2"].value},
                              }
            vary_dict      = {0:{'type':'sersic', 'mu_0':False, 'r_0':False, 'n':False},
                              1:{'type':'exp_disk', 'mu_0':False, 'h':False},
                              2:{'type':'exp_disk', 'mu_0':True, 'h':False}}
            i0=3
            if (fit_bar_rings):
                if (bar):
                    components_tmp[i0] = {'type':'bar', 'mu_0':result.params[f"mu_0{3}"].value, 'r':result.params[f"r{3}"].value, 'h':result.params[f"h{3}"].value}
                    vary_dict[i0]      = {'type':'bar', 'mu_0':True, 'r':False, 'h':False}
                    for i in range(n_rings):
                        components_tmp[i0+1+i] = {'type':'ring', 'mu_0':result.params[f"mu_0{4+i}"].value, 'r':result.params[f"r{4+i}"].value, 'w':result.params[f"w{4+i}"].value}
                        vary_dict[i0+1+i]      = {'type':'ring', 'mu_0':False, 'r':False, 'w':False}
                else:
                     for i in range(n_rings):
                        components_tmp[i0+i] = {'type':'ring', 'mu_0':result.params[f"mu_0{3+i}"].value, 'r':result.params[f"r{3+i}"].value, 'w':result.params[f"w{3+i}"].value}
                        vary_dict[i0+i]      = {'type':'ring', 'mu_0':False, 'r':False, 'w':False} 

            params_tmp = fill_params(components=components_tmp, vary_dict=vary_dict)   
            # params_tmp["h1"].min = 3.*r_hl
            # params_tmp["h1"].max = 2.5*np.log10(np.e)/slope2
            params_tmp["h2"].min = 2.5*np.log10(np.e)/slope2*0.2
            params_tmp["h2"].max = 2.5*np.log10(np.e)/slope2*1.8
            params_tmp["mu_02"].min = intercept2-0.1
            params_tmp["mu_02"].max = intercept2+0.1
            minner_tmp = Minimizer(fcn2min, params_tmp, fcn_args=(components_tmp, r_kpc, intens, intens_err, pixsize, scale_kpc, zeropoint, psf),
                            nan_policy='omit')
            result = minner_tmp.minimize(method='leastsq')

    model = make_model(result.params, components_tmp, pixsize, scale_kpc, zeropoint, r_kpc, psf, return_hr=True)
    export_result(objname, 
                  intens_in, intens_err_in, (~np.isnan(intens)).astype(int), psf, model, components_tmp, result.params, result.chisqr, result.nfree,
                  zeropoint, pixsize, scale_kpc, 
                  z, r_hl, 
                  H, omega_m, ell,
                  result.bic,
                  f"{dirout}/{outfname}.fits")

    if (plot):
        plot_decomposition(f"{dirout}/{outfname}.fits", cutout_file, f"{plot_dir}{objname}.pdf")

def plot_decomposition(params_file, image_file, outfname):

    p = fits.open(params_file)[1].data
    f = plt.figure(figsize=[14, 6])
    ax_main = f.add_axes([0.1, 0.3, 0.5, 0.58])
    ax_resid = f.add_axes([0.1, 0.1, 0.5, 0.175])
    ax_image = f.add_axes([0.62, 0.1, 0.343, 0.8])
    xlim = [-0.5, np.nanmax(p["r_prof"][0])*1.1]
    ax_main.plot(p["r_prof"][0], p["mag_prof"][0], '.k')
    ax_main.plot(p["r_model"][0], p["mag_model"][0])
    for i in range(len(p["mag_comps_conv"][0])):
        ax_main.plot(p["r_model"][0], p["mag_comps_conv"][0][i])
    # ax_main.plot(r_kpc, slope*r_kpc+intercept, 'k--')
    # if (fit_2_disks):
    #     ax_main.plot(r_kpc, slope2*r_kpc+intercept2, 'r--')
 
    ax_main.fill_between(p["r_prof"][0], min(30, max(p["mag_prof"][0]) + 1.), np.nanmin(p["mag_prof"][0])-0.5, where=(p["pixmask"][0]==0), color='grey', alpha=0.5)
    ax_main.fill_between(p["r_prof"][0], p["mag_prof_err_plus"][0], p["mag_prof_err_minus"][0], color='gray', alpha=0.7)
    # ax_main.fill_betweenx(y=[min(30, max(mag) + 1.), np.nanmin(mag)-0.5], x1=(r_kpc[fit_idx_max]+r_kpc[fit_idx_max])/2., x2=xlim[1], color='grey', alpha=0.5)
    ax_main.set_ylim(min(30, max(p["mag_prof"][0]) + 1.), np.nanmin(p["mag_prof"][0])-0.5)
    ax_main.set_xlim(xlim[0], xlim[1])

    ax_resid.plot(p["r_prof"][0], p["mag_model_pix"][0]-p["mag_prof"][0],  '.k', label='Data', zorder=0)
    ax_resid.axhline(0.0, ls='--', color='grey')
    ax_resid.set_ylim(-1, 1)
    ax_resid.set_xlim(xlim[0], xlim[1])
    ax_resid.set_ylabel('Residuals')
    ax_resid.set_xlabel("R, kpc")
    ax_main.set_title(p["name"][0])
    im = np.flip(mpimg.imread(image_file), axis=0)
    ax_image.imshow(im)
    ax_image.axis("off")
    # ax_main.legend()
    # plt.show()
    f.savefig(outfname)
    f.clf()
