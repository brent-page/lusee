import numpy as np
import healpy as hp
from time import time as timer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cmx
import pygdsm
from pygdsm import GlobalSkyModel, GlobalSkyModel2016
import spiceypy as spice
from datetime import datetime
from datetime import timedelta
from scipy.interpolate import interp1d

# gsm
gsm = GlobalSkyModel(interpolation = 'cubic')
gsm16 = GlobalSkyModel2016()


# NAIF codes
MOON = 301
EARTH = 399
JUPITER = 5
SATURN = 6
URANUS = 7
NEPTUNE = 8
SUN = 10


def lat_lon_to_rec(lat, lon):
    return np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)


def rec_to_lat_lon(x, y, z):
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return lat, lon


def rotx(angle):
    rot_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return rot_matrix


def roty(angle):
    rot_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return rot_matrix


def rotz(angle):
    rot_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return rot_matrix


# center of schrodinger basin in selenographic coordinates
# https://wma.wmflabs.org/iframe.html?-75_132_0_0_en_2_en_-75_132&globe=Moon&page=Schr%C3%B6dinger%20(crater)&lang=en
SCHROD_LAT = -75 * np.pi / 180
SCHROD_LON = 132.0 * np.pi / 180
SCHROD_OUTWARD_NORMAL = lat_lon_to_rec(SCHROD_LAT, SCHROD_LON)

# this rotation matrix can be used in lieu of the fns local_hour_from_azimuth_altitude and declination_from_azimuth_altitude; see the fn 'transformation test'
# local coordinates here are [cos(h)cos(-A), cos(h)sin(-A), sin(h)]
# A: azimuth measured west from south (by convention)
# h: altitude
# the MOON_ME coordinate axes can be rotated into the local coordinate axes via a rotation by (np.pi/2 - SCHROD_LAT) about the MOON_ME_Y axis, followed by a rotation by SCHROD_LON about the MOON_ME_Z axis.  Hence, the matrix below transforms a vector that is expressed in the local coordinate system into the same vector expressed in the MOON_ME coordinate system
local_to_moon_coords = np.matmul(rotz(SCHROD_LON), roty(np.pi / 2 - SCHROD_LAT))


def local_hour_from_azimuth_altitude(A_azimuth, h_altitude, phi_latitude):
    return np.arctan2(
        np.sin(A_azimuth) * np.cos(h_altitude),
        np.cos(A_azimuth) * np.sin(phi_latitude) * np.cos(h_altitude)
        + np.sin(h_altitude) * np.cos(phi_latitude),
    )

def declination_from_azimuth_altitude(A_azimuth, h_altitude, phi_latitude):
    return np.arcsin(
        np.sin(phi_latitude) * np.sin(h_altitude)
        - np.cos(phi_latitude) * np.cos(h_altitude) * np.cos(A_azimuth)
    )


def transformation_test():
    # arbitrary arc on the sky
    N = 100
    A_azimuth = np.linspace(np.pi / 3, 2 * np.pi, N)
    h_altitude = np.linspace(np.pi / 8, np.pi / 2, N)
    arc_local_coords = np.stack(lat_lon_to_rec(h_altitude, -A_azimuth)).T
    arc_moon_coords = np.matmul(local_to_moon_coords, arc_local_coords.T).T

    declination_method_1, right_ascension_method_1 = rec_to_lat_lon(*arc_moon_coords.T)

    declination_method_2 = declination_from_azimuth_altitude(
        A_azimuth, h_altitude, SCHROD_LAT
    )
    local_hour = local_hour_from_azimuth_altitude(A_azimuth, h_altitude, SCHROD_LAT)
    right_ascension_method_2 = SCHROD_LON - local_hour
    # make r.a. range from -pi to pi
    idxs = np.abs(right_ascension_method_2) > np.pi
    right_ascension_method_2[idxs] -= (
        np.sign(right_ascension_method_2[idxs]) * 2 * np.pi
    )

    plt.figure()
    plt.plot(declination_method_1, label="declination_method_1")
    plt.plot(declination_method_2, "--", label="declination_method_2")
    plt.plot(right_ascension_method_1, label="right_ascension_method_1")
    plt.plot(right_ascension_method_2, "--", label="right_ascension_method_2")
    plt.legend()


def load_kernels():
    # bsp: binary spk (spacecraft and planet kernel), positions of major solar system bodies (including the moon), https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    spice.furnsh("./lusee_kernels/de440s.bsp")

    # three kernels below: moon orientation (https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/)
    # bpc: binary pck (planetary constants (e.g. planet shape and orientation) kernel)
    spice.furnsh("./lusee_kernels/moon_pa_de421_1900-2050.bpc")
    # tf: text frame kernel
    spice.furnsh("./lusee_kernels/moon_080317.tf")
    spice.furnsh("./lusee_kernels/moon_assoc_me.tf")

    # tls: text leap second kernel
    spice.furnsh("./lusee_kernels/naif0012.tls")


# get the position of 'body' relative to the moon in moon mean-earth (MOON_ME) coordinates
def get_pos(body, times, ref = "MOON_ME"):
    body_pos = np.zeros((times.size, 3))
    for i, time in enumerate(times):
        body_pos[i] = spice.spkezp(
            targ=body, et=spice.datetime2et(time), ref=ref, abcorr="LT", obs=MOON
        )[0]
    body_pos = body_pos / ((np.sum(body_pos ** 2, axis=1)) ** (1 / 2))[:, np.newaxis]
    return body_pos


def get_altitude(body, times):
    body_pos = get_pos(body, times)
    body_altitude = (
        90 - np.arccos(np.matmul(body_pos, SCHROD_OUTWARD_NORMAL)) * 180 / np.pi
    )
    return body_altitude

def get_set_rise_idxs(below_horizon):
    set_idxs = below_horizon & ~np.roll(below_horizon, 1)
    rise_idxs = below_horizon & ~np.roll(below_horizon, -1)

    set_idxs[[0, -1]] = False
    rise_idxs[[0, -1]] = False

    set_idxs = np.flatnonzero(set_idxs)
    rise_idxs = np.flatnonzero(rise_idxs)

    # require full nights
    if np.min(rise_idxs) < np.min(set_idxs):
        rise_idxs = rise_idxs[1:]
    if np.max(set_idxs) > np.max(rise_idxs):
        set_idxs = set_idxs[:-1]

    return set_idxs, rise_idxs

# print out all lunar nights that overlap with this year/month
def print_lunar_night(year, month):
    utc_times = np.arange(
        datetime(year, month, 1) - timedelta(days=20),
        datetime(year, month, 1) + timedelta(days=50),
        timedelta(hours=1),
    ).astype(datetime)
    sun_altitude = get_altitude(SUN, utc_times)
    set_idxs, rise_idxs = get_set_rise_idxs(sun_altitude < 0)
    for set_time, rise_time in zip(utc_times[set_idxs], utc_times[rise_idxs]):
        if not ((set_time.month == month) or (rise_time.month) == month):
            continue
        else:
            print("night:")
            print("      start: " + str(set_time))
            print("      end: " + str(rise_time))
            print("")


def get_lunar_nights(year, delta_hour_search=1, delta_min_times=15):
    utc_times = np.arange(
        datetime(year, 1, 1), datetime(year, 12, 31), timedelta(hours=1)
    ).astype(datetime)
    sun_altitude = get_altitude(SUN, utc_times)
    set_idxs, rise_idxs = get_set_rise_idxs(sun_altitude < 0)
    nights = []
    for set_time, rise_time in zip(utc_times[set_idxs], utc_times[rise_idxs]):
        nights.append(
            np.arange(set_time, rise_time, timedelta(minutes=delta_min_times)).astype(
                datetime
            )
        )
    return nights


# get galactic latitude and longitude of outward normal from schrodinger basin
def get_pointing(utc_times):
    pointing = np.zeros((utc_times.size, 3))
    for i, time in enumerate(utc_times):
        rot = spice.pxform("MOON_ME", "GALACTIC", spice.datetime2et(time))
        pointing[i] = np.matmul(rot, SCHROD_OUTWARD_NORMAL)
    return rec_to_lat_lon(*pointing.T)


# plot the track of the schrod basin outward normal on the sky over time
def plot_day(utc_times, frequency, map_nside=512):
    lat, lon = get_pointing(utc_times)

    skymap = gsm.generate(frequency)
    skymap = hp.ud_grade(skymap, map_nside)
    skymap = np.log10(skymap)

    hp.mollview(
        skymap,
        coord="G",
        title="{} {:.0f} MHz, basemap: {}".format(gsm.name, frequency, gsm.basemap),
    )
    hp.projplot(np.pi / 2 - lat, lon, "wx")


# get vectors pointing to map pixels
def get_map_pixel_local_vecs(utc_times, beam_idxs=None, map_nside=512, nest=False):

    galactic_to_moon_rots = np.zeros((utc_times.size, 3, 3))
    for i, time in enumerate(utc_times):
        galactic_to_moon_rots[i] = spice.pxform(
            "GALACTIC", "MOON_ME", spice.datetime2et(time)
        )

    galac_vecs = np.stack(
        hp.pix2vec(map_nside, np.arange(hp.nside2npix(map_nside)), nest=nest)
    ).T

    if not (beam_idxs is None):
        galac_vecs = galac_vecs[beam_idxs]
        # transform a different set of pixel positions for each time
        # t: time, p: pixel
        local_vecs = np.einsum("ij,tjl,tpl->tpi", local_to_moon_coords.T, galactic_to_moon_rots, galac_vecs)
        
    else:
        # transform the positions of all pixels for each time
        local_vecs = np.einsum("ij,tjl,pl->tpi", local_to_moon_coords.T, galactic_to_moon_rots, galac_vecs)

    return galac_vecs, local_vecs


# to figure out which pixels are in the beam, look at a downsampled version of the map.  pixels in the downsampled map are groups of pixels from the higher res map.  if ( (the beam strength at the center of a downsampled pixel is > beam_threshold) and (the downsampled pixel center is above the horizon) ), accept all of its child pixels as being in the beam
# ds: downsampled
# fs: fullsampled
def get_beam_pixels(
    utc_times, local_vecs_ds, NS_beam_stdev, EW_beam_stdev, beam_threshold, fs_map_nside
):
    ds_map_nside = hp.npix2nside(local_vecs_ds.shape[1])
    ds_map_npix = hp.nside2npix(ds_map_nside)
    fs_map_npix = hp.nside2npix(fs_map_nside)
    npix_ratio = fs_map_npix // ds_map_npix
    # mapping of downsampled to fullsampled pixel numbers in the 'nest' pixel numbering scheme
    ds_to_fs = np.arange(fs_map_npix).reshape(ds_map_npix, npix_ratio)

    above_horizon = (local_vecs_ds[..., 2] > -0.05) 

    beam_weights = np.zeros(local_vecs_ds.shape[:-1])
    np.exp(
        -local_vecs_ds[..., 0] ** 2 / (2 * NS_beam_stdev ** 2)
        -local_vecs_ds[..., 1] ** 2 / (2 * EW_beam_stdev ** 2),
        out = beam_weights,
        where = above_horizon
        )

    sorted_beam_idxs = np.argsort(beam_weights, axis = 1)[:, ::-1]
    beam_weights = np.take_along_axis(beam_weights, sorted_beam_idxs, axis = 1)

    # for einsum, want to have the same number of pixels at each time
    num_ds_pixels_in_beam = np.max(np.sum(beam_weights > beam_threshold, axis = 1))
    beam_idxs = sorted_beam_idxs[:, :num_ds_pixels_in_beam]

    fs_in_beam_idxs = ds_to_fs[beam_idxs].reshape(-1, ds_to_fs.shape[1] * beam_idxs.shape[1])
    return hp.nest2ring(fs_map_nside, fs_in_beam_idxs)


# utc_times: np array of datetimes
# freqs: np array, MHz
# outline_sigma: beam gets cut-off (or tapered) at the outline_sigma contour of the beam
def time_freq_K(
        utc_times,
        freqs,
        NS_20MHz_beam_stdev_degr=5,
        EW_20MHz_beam_stdev_degr=5,
        widest_beam_freq=4,
        map_nside=512, 
        time_chunk = 100,
        outline_sigma = 3,
        use_envelope = True,
        taper_alpha = 0.1,
        horizon_taper = 5,
        plot = False, 
        verbose=False):

    NS_20MHz_beam_stdev = np.sin(NS_20MHz_beam_stdev_degr * np.pi/180)
    EW_20MHz_beam_stdev = np.sin(EW_20MHz_beam_stdev_degr * np.pi/180)

    # downsampled pixel positions 
    ds_map_nside = 64
    _, local_vecs_ds = get_map_pixel_local_vecs(
        utc_times, map_nside=ds_map_nside, nest=True
    )

    # find all pixels within the outline
    threshold = np.exp(-outline_sigma**2 / 2)

    # looking at downsampled map to get an idea of which pixels to sum over
    widest_beam_idxs = get_beam_pixels(
        utc_times,
        local_vecs_ds,
        NS_20MHz_beam_stdev * (20/widest_beam_freq),
        EW_20MHz_beam_stdev * (20/widest_beam_freq),
        threshold * 0.8,
        fs_map_nside=map_nside,
    )

    skymaps = gsm.generate(freqs)
    if verbose:
        print ("Generated maps nside = ", hp.get_nside(skymaps))
    if not (map_nside == hp.get_nside(skymaps)):
        skymaps = hp.ud_grade(skymaps, map_nside)
    if (freqs.size == 1):
        skymaps = skymaps[None, :]

    KK = np.zeros((utc_times.size, freqs.size))

    for ti_start in range(0,utc_times.size, time_chunk):
        ti_end = min(ti_start+time_chunk,utc_times.size)
        if verbose:
            print (f"     ...time {ti_start}:{ti_end}.") 

        # get local vecs for widest beam
        _, local_vecs = get_map_pixel_local_vecs(utc_times[ti_start:ti_end], widest_beam_idxs[ti_start:ti_end,:], map_nside)
        
        above_horizon = local_vecs[..., 2] > 0
        local_vecs_sq = local_vecs ** 2
        del local_vecs

        for i, freq in enumerate(freqs):
            if verbose:
                print (f" Integrating sky at {freq}MHz...")

            NS_beam_stdev = NS_20MHz_beam_stdev * 20/freq
            EW_beam_stdev = EW_20MHz_beam_stdev * 20/freq

            beam_weights = np.zeros(local_vecs_sq.shape[:-1])

            # local_vecs gets squared above
            np.exp(
                    -local_vecs_sq[..., 0] / (2 * NS_beam_stdev ** 2)
                    -local_vecs_sq[..., 1] / (2 * EW_beam_stdev ** 2),
                    out = beam_weights,
                    where = above_horizon
            )
            beam_weights[beam_weights < threshold] = np.nan

            if use_envelope:
                NS_envelope_outline = outline_sigma * NS_beam_stdev
                EW_envelope_outline = outline_sigma * EW_beam_stdev

                beam_envelope = get_beam_envelope(local_vecs_sq, NS_envelope_outline, EW_envelope_outline, alpha = taper_alpha, horizon_taper = horizon_taper)
                beam_weights *= beam_envelope

            KK[ti_start:ti_end, i] = np.nansum(skymaps[i, widest_beam_idxs[ti_start:ti_end]] * beam_weights, axis=1) / np.nansum(
                beam_weights, axis=1)

            if np.any(np.isnan(KK)):
                print ("Waterfall has nans, time to die.")
                stop()

    if plot:
        plt.figure()
        plt.pcolormesh(utc_times, freqs, KK.T, shading="nearest", norm=mcolors.LogNorm())
        cbar = plt.colorbar()
        plt.ylabel("freq (MHz)")
        cbar.set_label("Temp. (K)")
        plt.title(
            "RJ Temp, NS 20 MHz sigma: {:.0f}°, EW 20 MHz sigma: {:.0f}°".format(
                NS_20MHz_beam_stdev_degr,
                EW_20MHz_beam_stdev_degr,
            )
        )

    return KK

def create_reference():
    utc_times = np.array([datetime(2024, 3, 21, 21), datetime(2024, 3, 22, 21)]).astype(datetime)
    KK = time_freq_K(
        utc_times,
        freqs=np.arange(10, 16, 5),
        NS_20MHz_beam_stdev_degr=5,
        EW_20MHz_beam_stdev_degr=5,
        map_nside=512,
        use_envelope = False
    )
    return KK
#     np.savez('waterfall_ref.npz', KK = KK)


def create_reference():
    utc_times = np.array([datetime(2024, 3, 21, 21), datetime(2024, 3, 22, 21)]).astype(datetime)
    KK = time_freq_K(
        utc_times,
        freqs=np.arange(10, 16, 5),
        NS_20MHz_beam_stdev_degr=5,
        EW_20MHz_beam_stdev_degr=5,
        map_nside=512,
    )
    return KK
#     np.savez('waterfall_ref.npz', KK = KK)



def drive(minutes = 240, threshold = .01, NS = 60, EW = 5, nside = 128, time_chunk = 20, plot = False):
    sta = timer()
    utc_times = np.arange(
        datetime(2024, 3, 21, 21), datetime(2024, 4, 5, 11), timedelta(minutes=minutes)
    ).astype(datetime)
    KK = time_freq_K(
        utc_times,
#         freqs=np.arange(10, 71, 10),
        freqs=np.arange(20, 50, .5),
        NS_20MHz_beam_stdev_degr=NS,
        EW_20MHz_beam_stdev_degr=EW,
        map_nside=nside,
        time_chunk=time_chunk,
        plot = plot
    )
    sto = timer()
    print(sto - sta)


# utc_time: datetime
def plot_beam(utc_time, NS_beam_stdev_degr, EW_beam_stdev_degr, map_nside=512, outline_sigma = 3, use_envelope = False, taper_alpha = 0.1, horizon_taper = 5):

    NS_beam_stdev = np.sin(NS_beam_stdev_degr * np.pi/180)
    EW_beam_stdev = np.sin(EW_beam_stdev_degr * np.pi/180)

    utc_time = np.array([utc_time])

    # downsampled pixel positions 
    ds_map_nside = 64
    _, local_vecs_ds = get_map_pixel_local_vecs(
        utc_time, map_nside=ds_map_nside, nest=True
    )

    threshold = np.exp(-outline_sigma**2 / 2)

    beam_idxs = get_beam_pixels(
        utc_time, local_vecs_ds, NS_beam_stdev, EW_beam_stdev, threshold, fs_map_nside=map_nside
    )

    galac_vecs, local_vecs = get_map_pixel_local_vecs(
        utc_time, beam_idxs, map_nside=map_nside, nest=False
    )

    galac_vecs = galac_vecs[0]
    local_vecs = local_vecs[0]

    galac_lats, galac_lons = rec_to_lat_lon(*galac_vecs.T)

    above_horizon = local_vecs[..., 2] > 0
    beam_weights = np.zeros(local_vecs.shape[:-1])
    np.exp(
        -local_vecs[..., 0] ** 2 / (2 * NS_beam_stdev ** 2)
        -local_vecs[..., 1] ** 2 / (2 * EW_beam_stdev ** 2),
        out = beam_weights,
        where = above_horizon
    )
    if use_envelope:
        NS_envelope_outline = outline_sigma * NS_beam_stdev
        EW_envelope_outline = outline_sigma * EW_beam_stdev

        beam_envelope = get_beam_envelope(local_vecs ** 2, NS_envelope_outline, EW_envelope_outline, alpha = taper_alpha, horizon_taper = horizon_taper)
        beam_weights *= beam_envelope

    frequency = 20
    skymap = gsm.generate(frequency)
    skymap = hp.ud_grade(skymap, map_nside)
    skymap = np.log10(skymap)

    hp.mollview(
        skymap,
        coord="G",
        title="{} {:.0f} MHz, basemap: {}".format(gsm.name, frequency, gsm.basemap),
    )
    hp.projscatter(
        theta=np.pi / 2 - galac_lats,
        phi=galac_lons,
        lonlat=False,
        c=beam_weights,
        cmap=plt.cm.Blues,
        s=0.01 * (512 / map_nside) ** 2,
    )


def get_delta_T (utc_times, frequency, wfall, return_inv2 = False):
    deltaT = ((utc_times[-1]-utc_times[0])/(len(utc_times)-1)).seconds
    deltaf = (frequency[-1]-frequency[0])/(len(frequency)-1)
    #print (f"Delta T: {deltaT} Delta f: {deltaf}MHz")
    inv2 = ((deltaf*1e6)*deltaT/wfall**2).sum(axis=0)
    if return_inv2:
        return inv2
    return np.sqrt(1/inv2)

def get_modes (wfall):
    mean = wfall.mean(axis=0)
    mdiv = wfall/mean[None,:]-1
    cov = np.cov(mdiv,rowvar=False)
    eva,eve = np.linalg.eig(cov)
    return mean, eva, eve

def get_signal (freq):
    # Returns expected signal in K on the given frequency array
    da = np.load('waterfalls/signal.npz')
    nu = da['nu']
    Tsig = da['Tsig']
    return interp1d(nu,Tsig,kind='cubic')(freq)

    
# alpha: in an arc extending from zenith to the envelope outline, alpha approximately is the fraction of the arc occupied by the taper (in the case that the beam doesn't reach the horizon)
# horizon_taper (degrees): taper width in the case that the beam reaches the horizon
# points_in_taper: at minimum, should be a few times the number of timestamps it takes for a pixel to move through the taper
def get_beam_envelope(local_vecs_sq, NS_outline, EW_outline, alpha = 0.1, horizon_taper = 5, points_in_taper = 500):

    horizon_alpha = horizon_taper/90

    points_in_beam_envelope = int( (points_in_taper) / alpha )
    points_in_horizon_envelope = int( (points_in_taper) / horizon_alpha )
    cosine_taper = 1/2 * (1 + np.cos(np.pi * np.arange(points_in_taper) / (points_in_taper - 1)) )

    beam_envelope = np.ones(points_in_beam_envelope)
    horizon_envelope = np.ones(points_in_horizon_envelope)

    beam_envelope[-points_in_taper:] = cosine_taper
    horizon_envelope[-points_in_taper:] = cosine_taper

    # index based on where local_vec is in relation to the beam outline
    # if EW_outline = 3 * (beam EW stdev) and NS_outline = 3 * (beam NS stdev), then the envelope outline will be the beam's 3 sigma contour (strength = exp(-3**2 / 2))
    beam_envelope_index = np.round( 
            np.sqrt( 
                local_vecs_sq[..., 0] / (NS_outline**2) 
                + local_vecs_sq[..., 1] / (EW_outline**2) 
                ) * (points_in_beam_envelope - 1)
            ).astype(int)

    def not_more_than_one(x):
        x[x > 1.0] = 1.0
        return x
    # index based on where local_vec is in relation to the horizon
    horizon_envelope_index = np.rint(
            np.arcsin(
                not_more_than_one(
                    np.sqrt(local_vecs_sq[..., 0] + local_vecs_sq[..., 1])
                    )
                ) / (np.pi / 2) * (points_in_horizon_envelope - 1)
            ).astype(int)

    beam_envelope_index[beam_envelope_index >= points_in_beam_envelope] = points_in_beam_envelope - 1
    beam_envelope_vals = beam_envelope[beam_envelope_index]

    horizon_envelope_vals = horizon_envelope[horizon_envelope_index]

    # choose the more limiting taper
    return np.where(beam_envelope_vals < horizon_envelope_vals, beam_envelope_vals, horizon_envelope_vals)


global KK1, KK2
def beam_continuity_check():
    global KK1, KK2
    dt_minutes = 15
    utc_times = np.arange(datetime(2024, 3, 21, 21), datetime(2024, 4, 5, 11), timedelta(minutes = dt_minutes)).astype(datetime)[940:970]
    freqs = np.arange(10, 15, .1)
    maps = gsm.generate(freqs)

    logmaps = np.log(maps/maps[0])
    logfreqs = np.log(freqs/freqs[0])

    log_power_law_residual = lambda p, x, y: (y - (p[0] * x))

    spec_indices = np.zeros(maps.shape[1])
    mean_abs_residuals = np.zeros(maps.shape[1])
    for pixel_num in range(maps.shape[1]):
        fit = optimize.leastsq(log_power_law_residual, x0 = [-2.5], args = (logfreqs, logmaps[:, pixel_num]))
        spec_indices[pixel_num] = fit[0][0]
        mean_abs_residuals[pixel_num] = np.mean(np.absolute(log_power_law_residual(fit[0], logfreqs, logmaps[:, pixel_num])**2))

def power_law_plot():
    freqs = np.arange(10, 15, .1)
    maps = gsm.generate(freqs)

    logmaps = np.log(maps/maps[0])
    logfreqs = np.log(freqs/freqs[0])

    random = np.random.randint(0, maps.shape[1], 100)
    plt.figure()
    for num in random:
        plt.plot(logfreqs, logmaps[:, num])

    plt.xlabel(r'log($\nu$/10MHz)')
    plt.ylabel(r'log($T/T_{10MHz}$)')
    plt.title(r'10-15 MHz pyGDSM T vs $\nu$ for 100 random pixels')

    KK1 = time_freq_K(utc_times, freqs, NS_at_20MHz, EW_at_20MHz, 512, use_envelope = True, verbose = True, taper_alpha = .1, horizon_taper = 5, outline_sigma = 3)
    KK2 = time_freq_K(utc_times, freqs, NS_at_20MHz, EW_at_20MHz, 512, use_envelope = False, verbose = True, outline_sigma = 3)

    KK1 -= np.mean(KK1)
    KK2 -= np.mean(KK2)

#     fft_freqs = np.fft.rfftfreq(utc_times.size, dt_minutes/60)
#     fig, axs = plt.subplots(3, sharex = True, figsize = (12, 6))
#     for KK, label in zip((KK1, KK2), ('with envelope', 'without envelope')):
#         KK_fft = np.fft.rfft(KK, axis = 0)
#         for i, freq in enumerate(freqs):
#             axs[i].plot(fft_freqs, np.absolute(KK_fft[:, i])/np.mean(KK[:, i]), label = label)
# 
#     for ax in axs:
#         ax.legend()
#         ax.set_yscale('log')
#     axs[0].set_xlim(1.75, 2)
#     axs[-1].set_xlabel('Freq (1/Hr)')


def plot_enveloped_beam(NS, EW, outline_sigma, taper_alpha, horizon_taper):
    NS = np.sin(NS * np.pi/180)
    EW = np.sin(EW * np.pi/180)

    x = np.linspace(-1, 1, 1000)
    y = np.linspace(-1, 1, 1000)
    X, Y = np.meshgrid(x, y)

    norm = np.sqrt(X**2 + Y**2)

    X[norm > 1]/= norm[norm > 1] * 1.0001
    Y[norm > 1]/= norm[norm > 1] * 1.0001

    flat_coords = np.zeros((X.size, 2))
    flat_coords[:, 0] = X.flatten()
    flat_coords[:, 1] = Y.flatten()

    beam_weights = np.exp(
            -X**2 / (2 * NS**2)
            -Y**2 / (2 * EW**2)
            )
    beam_envelope = get_beam_envelope(flat_coords**2, NS * outline_sigma, EW * outline_sigma, alpha = taper_alpha, horizon_taper = horizon_taper).reshape(-1, x.size)

    fig, axs = plt.subplots(1, 3, figsize = (15, 5))
    axs[0].pcolormesh(X, Y, beam_weights, shading = 'nearest')
    axs[1].pcolormesh(X, Y, beam_envelope, shading = 'nearest')
    axs[2].pcolormesh(X, Y, beam_weights * beam_envelope, shading = 'nearest')

    for ax in axs:
        ax.set_aspect('equal')


def sky_spec_indices(mask = False):

    spec = spec_indices.copy()
    resid = mean_abs_residuals.copy()
    if mask:
#         sel = ((spec < -3) | (spec > -2))
        sel = (resid)**(1/2) > 0.01
        spec[sel] = np.nan
        resid[sel] = np.nan
    hp.mollview((resid)**(1/2), title = 'mean absolute power law fit residuals \n frequency range: 10-15 MHz \n' + r'(mean over frequency)$\left[|\log\left(\frac{T}{T_{10MHz}}\right) - \beta_{fit}\log\left(\frac{\nu}{10MHz}\right)|\right]$' + '\n masked above 0.01')
    hp.mollview(spec, title = 'spectral indices between 10 and 15 MHz')


def get_signal (freq):
    # Returns expected signal in K on the given frequency array
    da = np.load('waterfalls/signal.npz')
    nu = da['nu']
    Tsig = da['Tsig']
    return interp1d(nu,Tsig,kind='cubic')(freq)

    
