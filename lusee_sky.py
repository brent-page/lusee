# from importlib import reload as rl
# import xarray as xr
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

# gsm
gsm = GlobalSkyModel()
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
def get_pos(body, times):
    body_pos = np.zeros((times.size, 3))
    for i, time in enumerate(times):
        body_pos[i] = spice.spkezp(
            targ=body, et=spice.datetime2et(time), ref="MOON_ME", abcorr="LT", obs=MOON
        )[0]
    body_pos = body_pos / ((np.sum(body_pos ** 2, axis=1)) ** (1 / 2))[:, np.newaxis]
    return body_pos


def get_altitude(body, times):
    body_pos = get_pos(body, times)
    body_altitude = (
        90 - np.arccos(np.matmul(body_pos, SCHROD_OUTWARD_NORMAL)) * 180 / np.pi
    )
    return body_altitude


def get_sun_earth_from_horizon():
    utc_times = np.arange(
        datetime(2024, 1, 1), datetime(2024, 12, 31, 23, 59), timedelta(minutes=15)
    ).astype(datetime)

    earth_altitude = get_altitude(EARTH, utc_times)
    sun_altitude = get_altitude(SUN, utc_times)

    sun_earth_below_horizon = (earth_altitude < 0) & (sun_altitude < 0)
    sun_earth_above_horizon = (earth_altitude > 0) & (sun_altitude > 0)

    start_both_below_idxs, end_both_below_idxs = get_set_rise_idxs(
        sun_earth_below_horizon
    )
    start_both_above_idxs, end_both_above_idxs = get_set_rise_idxs(
        sun_earth_above_horizon
    )

    below_durations = utc_times[end_both_below_idxs] - utc_times[start_both_below_idxs]
    above_durations = utc_times[end_both_above_idxs] - utc_times[start_both_above_idxs]

    above_label = np.array(["above"] * start_both_above_idxs.size)
    below_label = np.array(["below"] * start_both_below_idxs.size)

    labels = np.hstack((below_label, above_label))
    start_idxs = np.hstack((start_both_below_idxs, start_both_above_idxs))
    end_idxs = np.hstack((end_both_below_idxs, end_both_above_idxs))

    durations = np.hstack((below_durations, above_durations))
    sort_idxs = np.argsort(start_idxs)

    labels = labels[sort_idxs]
    start_idxs = start_idxs[sort_idxs]
    end_idxs = end_idxs[sort_idxs]
    durations = durations[sort_idxs]

    to_export2 = xr.Dataset(
        {
            "both above or both below horizon": (["idx"], labels),
            "start": (["idx"], utc_times[start_idxs]),
            "end": (["idx"], utc_times[end_idxs]),
            "duration": (["idx"], durations),
        }
    )
    with open("earth_sun_from_120_east.csv", "w") as new_file:
        new_file.write(to_export2.to_dataframe().to_csv(index_label=""))


def plot_inclinations_from_horizon():
    utc_times = np.arange(
        datetime(2024, 1, 1), datetime(2025, 12, 31, 23, 59), timedelta(minutes=15)
    ).astype(datetime)

    saturn_altitude = get_altitude(SATURN, utc_times)
    jupiter_altitude = get_altitude(JUPITER, utc_times)
    #     neptune_altitude = get_altitude(NEPTUNE, utc_times)
    #     uranus_altitude = get_altitude(URANUS, utc_times)
    sun_altitude = get_altitude(SUN, utc_times)

    #     all_below_horizon = ((saturn_altitude < 0) & (jupiter_altitude < 0) & (sun_altitude < 0) & (neptune_altitude < 0) & (uranus_altitude < 0))
    sun_jupiter_saturn_below_horizon = (
        (saturn_altitude < 0) & (jupiter_altitude < 0) & (sun_altitude < 0)
    )
    sun_below_horizon = sun_altitude < 0
    #     sun_jupiter_below_horizon = ((sun_altitude < 0) & (jupiter_altitude < 0))
    #     ura_nep_below_horizon = ((uranus_altitude < 0) & (neptune_altitude < 0))
    to_export = xr.Dataset(
        {
            "sun altitude (deg)": (["time"], sun_altitude),
            "jupiter altitude (deg)": (["time"], jupiter_altitude),
            "saturn altitude (deg)": (["time"], saturn_altitude),
            "sun below horizon?": (["time"], sun_altitude < 0),
            "jupiter below horizon?": (["time"], jupiter_altitude < 0),
            "saturn below horizon?": (["time"], saturn_altitude < 0),
            "all below horizon?": (["time"], sun_jupiter_saturn_below_horizon),
        },
        coords={"time": utc_times},
    )
    csv = to_export.to_dataframe().round(decimals=1).to_csv()
    with open("schrod_basin_sun_jup_sat_visibility.csv", "w") as new_file:
        new_file.write(csv)

    for below_horizon, name, label in zip(
        [
            sun_below_horizon,
            all_below_horizon,
            sun_jupiter_below_horizon,
            ura_nep_below_horizon,
        ],
        [
            "sun_radio_quiet_times",
            "sun_jup_sat_ura_nep_radio_quiet_times",
            "sun_jupiter_radio_quiet_times",
            "uranus_neptune_radio_quiet_times",
        ],
        [
            "sun",
            "sun, jupiter, saturn, uranus, neptune",
            "sun, jupiter",
            "uranus, neptune",
        ],
    ):
        radio_quiet_start_idxs, radio_quiet_end_idxs = get_set_rise_idxs(below_horizon)

        durations = utc_times[radio_quiet_end_idxs] - utc_times[radio_quiet_start_idxs]
        to_export2 = xr.Dataset(
            {
                label
                + " below horizon start": (["idx"], utc_times[radio_quiet_start_idxs]),
                label
                + " below horizon end": (["idx"], utc_times[radio_quiet_end_idxs]),
                "duration": (["idx"], durations),
            }
        )
        with open(name + ".csv", "w") as new_file:
            new_file.write(to_export2.to_dataframe().to_csv(index_label=""))


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


def get_planet_visible(planet_code):
    utc_times = np.arange(
        datetime(2024, 1, 1), datetime(2024, 12, 31, 23, 59), timedelta(minutes=15)
    ).astype(datetime)
    altitude = get_altitude(planet_code, utc_times)
    start_idxs, end_idxs = get_set_rise_idxs(altitude > 0)

    durations = utc_times[end_idxs] - utc_times[start_idxs]
    to_export2 = xr.Dataset(
        {
            "above horizon start": (["idx"], utc_times[start_idxs]),
            "above horizon end": (["idx"], utc_times[end_idxs]),
            "duration": (["idx"], durations),
        }
    )
    return to_export2


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
    hp.projplot(np.pi / 2 - lat, lon, "rx")


# get vectors pointing to map pixels
def get_map_pixel_local_vecs(utc_times, beam_idxs=None, map_nside=512, nest=False):

    galactic_to_moon_rots = np.zeros((utc_times.size, 3, 3))
    for i, time in enumerate(utc_times):
        galactic_to_moon_rots[i] = spice.pxform(
            "GALACTIC", "MOON_ME", spice.datetime2et(time)
        )
    galactic_to_local_rots = np.einsum(
        "ij,kjl->kil", local_to_moon_coords.T, galactic_to_moon_rots
    )

    galac_vecs = np.stack(
        hp.pix2vec(map_nside, np.arange(hp.nside2npix(map_nside)), nest=nest)
    ).T

    if not (beam_idxs is None):
        # transform a different set of pixel positions for each time
        galac_vecs = galac_vecs[beam_idxs]
        local_vecs = np.einsum("ijk,ilk->ilj", galactic_to_local_rots, galac_vecs)
    else:
        # transform the positions of all pixels for each time
        local_vecs = np.einsum("ijk,lk->ilj", galactic_to_local_rots, galac_vecs)

    return galac_vecs, local_vecs


# to figure out which pixels are in the beam, look at a downsampled version of the map.  pixels in the downsampled map are groups of pixels from the higher res map.  if ( (the beam strength at the center of a downsampled pixel is > beam_threshold) and (the downsampled pixel center is above the horizon) ), accept all of its child pixels as being in the beam
# ds: downsampled
# fs: fullsampled
def get_beam_pixels(
    utc_times, NS_beam_stdev, EW_beam_stdev, beam_threshold, fs_map_nside
):
    ds_map_nside = 64
    ds_map_npix = hp.nside2npix(ds_map_nside)
    fs_map_npix = hp.nside2npix(fs_map_nside)
    npix_ratio = fs_map_npix // ds_map_npix
    # mapping of downsampled to fullsampled pixel numbers in the 'nest' pixel numbering scheme
    ds_to_fs = np.arange(fs_map_npix).reshape(ds_map_npix, npix_ratio)

    _, local_vecs_ds = get_map_pixel_local_vecs(
        utc_times, map_nside=ds_map_nside, nest=True
    )
    beam_weights = np.exp(
        -local_vecs_ds[..., 0] ** 2 / (2 * NS_beam_stdev ** 2)
    ) * np.exp(-local_vecs_ds[..., 1] ** 2 / (2 * EW_beam_stdev ** 2))
    in_beam = (local_vecs_ds[..., 2] > -0.02) & (beam_weights > beam_threshold)
    not_in_beam_idxs = [np.flatnonzero(~in_beam_at_time) for in_beam_at_time in in_beam]
    in_beam_idxs = [np.flatnonzero(in_beam_at_time) for in_beam_at_time in in_beam]
    max_pixel_num = int(np.max(np.sum(in_beam, axis=1)) * npix_ratio)
    fs_in_beam_idxs = np.zeros((utc_times.size, max_pixel_num))
    for i, (this_in_beam, this_not_in_beam) in enumerate(
        zip(in_beam_idxs, not_in_beam_idxs)
    ):
        this_pixel_num = this_in_beam.size * npix_ratio
        fs_in_beam_idxs[i, :this_pixel_num] = hp.nest2ring(
            fs_map_nside, ds_to_fs[this_in_beam]
        ).flatten()

        # in order to have the same number of pixels at each time, add some throwaway pixels if necessary
        fs_in_beam_idxs[i, this_pixel_num:] = hp.nest2ring(
            fs_map_nside,
            ds_to_fs[
                this_not_in_beam[: (max_pixel_num - this_pixel_num) // npix_ratio]
            ],
        ).flatten()

    return fs_in_beam_idxs.astype(int)


# utc_times: np array of datetimes
# freqs: np array, MHz
def time_freq_K(
        utc_times,
        freqs,
        NS_20MHz_beam_stdev=np.sin(5 * np.pi / 180),
        EW_20MHz_beam_stdev=np.sin(5 * np.pi / 180),
        map_nside=512, plot = False,
        time_chunk = 100, verbose=False):

    threshold = 0.01


    KK = np.zeros((utc_times.size, freqs.size))
    for i, freq in enumerate(freqs):
        if verbose:
            print (f" Getting sky at {freq}MHz...")
        NS_beam_stdev = NS_20MHz_beam_stdev * (20 / freq)
        EW_beam_stdev = EW_20MHz_beam_stdev * (20 / freq)
        # looking at downsampled map to get an idea of which pixels to sum over
        beam_idxs = get_beam_pixels(
            utc_times,
            NS_beam_stdev,
            EW_beam_stdev,
            threshold,
            fs_map_nside=map_nside,
        )

        skymap = gsm.generate(freq)
        if not (map_nside == hp.get_nside(skymap)):
            skymap = hp.ud_grade(skymap, map_nside)
        for ti_start in range(0,utc_times.size, time_chunk):
            ti_end = min(ti_start+time_chunk,utc_times.size)
            if verbose:
                print (f"     ...time {ti_start}:{ti_end}.") 
            _, local_vecs = get_map_pixel_local_vecs(utc_times[ti_start:ti_end], beam_idxs[ti_start:ti_end,:], map_nside)
            below_horizon = local_vecs[..., 2] < 0
            local_vecs = local_vecs ** 2

            beam_weights = np.exp(-local_vecs[..., 0] / (2 * NS_beam_stdev ** 2)) * np.exp(
                -local_vecs[..., 1] / (2 * EW_beam_stdev ** 2)
            )
            #          pixels that are below the horizon can't be seen
            beam_weights[below_horizon] = 0
            KK[ti_start:ti_end, i] = np.sum(skymap[beam_idxs[ti_start:ti_end,:]] * beam_weights, axis=1) / np.sum(
                beam_weights, axis=1
            )

    if plot:
        plt.figure()
        plt.pcolormesh(utc_times, freqs, KK.T, shading="nearest", norm=mcolors.LogNorm())
        cbar = plt.colorbar()
        plt.ylabel("freq (MHz)")
        cbar.set_label("Temp. (K)")
        plt.title(
            "RJ Temp, NS 20 MHz sigma: {:.0f}°, EW 20 MHz sigma: {:.0f}°".format(
                np.arcsin(NS_20MHz_beam_stdev) * 180 / np.pi,
                np.arcsin(EW_20MHz_beam_stdev) * 180 / np.pi,
            )
        )

    return KK


def drive():
    sta = timer()
    utc_times = np.arange(
        datetime(2024, 3, 21, 21), datetime(2024, 4, 5, 11), timedelta(hours=2)
    ).astype(datetime)
    KK = time_freq_K(
        utc_times,
        freqs=np.arange(20, 51, 1),
        NS_20MHz_beam_stdev=np.sin(60 * np.pi / 180),
        EW_20MHz_beam_stdev=np.sin(5 * np.pi / 180),
        map_nside=128,
    )
    sto = timer()
    print(sto - sta)


# utc_time: datetime
def plot_beam(utc_time, NS_beam_stdev, EW_beam_stdev, map_nside=512):
    threshold = 0.01
    utc_time = np.array([utc_time])
    beam_idxs = get_beam_pixels(
        utc_time, NS_beam_stdev, EW_beam_stdev, threshold, fs_map_nside=map_nside
    )
    galac_vecs, local_vecs = get_map_pixel_local_vecs(
        utc_time, beam_idxs, map_nside=map_nside, nest=False
    )
    galac_vecs = galac_vecs[0]

    galac_lats, galac_lons = rec_to_lat_lon(*galac_vecs.T)

    beam_weights = np.exp(
        -local_vecs[0, :, 0] ** 2 / (2 * NS_beam_stdev ** 2)
    ) * np.exp(-local_vecs[0, :, 1] ** 2 / (2 * EW_beam_stdev ** 2))

    # pixels that are below the horizon can't be seen
    beam_weights[local_vecs[0, :, 2] < 0] = 0
    beam_sel = beam_weights > 0

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
        theta=np.pi / 2 - galac_lats[beam_sel],
        phi=galac_lons[beam_sel],
        lonlat=False,
        c=beam_weights[beam_sel],
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
