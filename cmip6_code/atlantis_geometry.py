"""
Atlantis geometry helpers for the East Antarctic model.

Reads the box polygons from the BGM file and the vertical layer structure from
the Atlantis initial-conditions file, so that CMIP6 data can be mapped onto
Atlantis boxes and layers by their real depth.

Atlantis layer convention (checked against EAtemp.nc / EAsalt.nc and
init_squid_230809.nc):
    z index 0                  = deepest water layer of the box
    z index numlayers[b] - 1   = surface layer of the box
    z index numlayers..wcnz-1  = layers that do not exist in this box (value 0)
    z index wcnz               = sediment layer
"""

import numpy as np
import netCDF4
from pyproj import Transformer


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Read box polygons from the BGM file ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def read_bgm_boxes(bgm_path):
    """
    Read box polygons, bottom depths and the map projection from a BGM file.

    Returns a dict with:
        projection : proj4 string of the BGM coordinates
        boxes      : list (ordered by box number) of dicts with
                     label, botz (m, negative down), area (m2),
                     x, y (polygon vertices in BGM projected metres)
    """
    projection = None
    n_boxes = None
    box_lines = {}

    with open(bgm_path) as bgm_file:
        for line in bgm_file:
            parts = line.split()
            if len(parts) == 0 or parts[0].startswith("#"):
                continue
            key = parts[0]
            if key == "projection":
                projection = " ".join(parts[1:])
            elif key == "nbox":
                n_boxes = int(parts[1])
            elif key.startswith("box") and "." in key:
                box_name, field = key.split(".", 1)
                box_number = int(box_name[3:])
                box_lines.setdefault(box_number, []).append((field, parts[1:]))

    if projection is None or n_boxes is None:
        raise ValueError(f"BGM file {bgm_path} has no 'projection' or 'nbox' line")
    if sorted(box_lines.keys()) != list(range(n_boxes)):
        raise ValueError(f"BGM file lists boxes {sorted(box_lines)} but nbox = {n_boxes}")

    boxes = []
    for box_number in range(n_boxes):
        box = {"number": box_number, "x": [], "y": []}
        for field, values in box_lines[box_number]:
            if field == "label":
                box["label"] = values[0]
            elif field == "botz":
                box["botz"] = float(values[0])
            elif field == "area":
                box["area"] = float(values[0])
            elif field == "vert":
                box["x"].append(float(values[0]))
                box["y"].append(float(values[1]))
        box["x"] = np.array(box["x"])
        box["y"] = np.array(box["y"])
        if len(box["x"]) < 3:
            raise ValueError(f"Box {box_number} has fewer than 3 vertices in {bgm_path}")
        boxes.append(box)

    return {"projection": projection, "boxes": boxes}


def make_lonlat_to_bgm_transformer(bgm_projection):
    """Transformer from geographic lon/lat (WGS84) to the BGM projected x/y."""
    return Transformer.from_crs("EPSG:4326", bgm_projection, always_xy=True)


def add_lonlat_to_boxes(bgm):
    """Add lon/lat vertices to every box (used for cropping and plotting)."""
    to_lonlat = Transformer.from_crs(bgm["projection"], "EPSG:4326", always_xy=True)
    for box in bgm["boxes"]:
        lon, lat = to_lonlat.transform(box["x"], box["y"])
        box["lon"] = np.mod(lon, 360.0)
        box["lat"] = lat
    return bgm


def domain_lonlat_bounds(bgm, buffer_degrees):
    """
    Lon/lat bounding box of all Atlantis boxes plus a buffer, with longitudes
    in 0-360. Fails if the domain straddles 0/360 E (not the case for East
    Antarctica, ~30-130 E), because the simple crop would then be wrong.
    """
    all_lon = np.concatenate([box["lon"] for box in bgm["boxes"]])
    all_lat = np.concatenate([box["lat"] for box in bgm["boxes"]])
    if all_lon.max() - all_lon.min() > 180.0:
        raise ValueError("Atlantis domain seems to straddle 0/360 E; cropping logic must be adapted")
    return {
        "lon_min": all_lon.min() - buffer_degrees,
        "lon_max": all_lon.max() + buffer_degrees,
        "lat_min": max(all_lat.min() - buffer_degrees, -90.0),
        "lat_max": min(all_lat.max() + buffer_degrees, 90.0),
    }


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## ~ Read the Atlantis vertical layer structure ~ ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def read_atlantis_layers(init_path, bgm=None):
    """
    Build the depth range of every Atlantis water layer from nominal_dz and
    numlayers in the Atlantis initial-conditions file.

    Returns a dict with:
        n_water_layers  : wcnz (10 for EA)
        sediment_index  : z index of the sediment layer (= wcnz)
        numlayers       : [n_boxes] number of water layers in each box
        layer_top       : [n_boxes, wcnz] top depth of each z index (m, positive down)
        layer_bottom    : [n_boxes, wcnz] bottom depth of each z index
        Unused layers have NaN depth.
    """
    with netCDF4.Dataset(init_path) as init_nc:
        init_nc.set_auto_mask(False)
        nominal_dz = np.array(init_nc["nominal_dz"][:], dtype=float)      # [b, z]
        dz_fill = init_nc["nominal_dz"]._FillValue
        numlayers = np.array(init_nc["numlayers"][0, :], dtype=int)      # [b]
        n_water_layers = int(init_nc.getncattr("wcnz"))

    n_boxes = nominal_dz.shape[0]
    layer_top = np.full((n_boxes, n_water_layers), np.nan)
    layer_bottom = np.full((n_boxes, n_water_layers), np.nan)

    for box_number in range(n_boxes):
        n_layers = numlayers[box_number]
        thickness = nominal_dz[box_number, :n_layers]
        if np.any(thickness == dz_fill) or np.any(thickness <= 0):
            raise ValueError(f"Box {box_number}: invalid nominal_dz {thickness}")
        # z index k (0 = deepest) sits below all layers k+1 ... n_layers-1
        for k in range(n_layers):
            layer_top[box_number, k] = thickness[k + 1:].sum()
            layer_bottom[box_number, k] = layer_top[box_number, k] + thickness[k]

    # The deepest layer should reach the BGM bottom depth
    if bgm is not None:
        for box in bgm["boxes"]:
            total_depth = layer_bottom[box["number"], 0]
            if not np.isclose(total_depth, -box["botz"], atol=1.0):
                raise ValueError(f"Box {box['number']}: layers reach {total_depth} m "
                                 f"but BGM botz is {box['botz']} m")

    return {
        "n_water_layers": n_water_layers,
        "sediment_index": n_water_layers,
        "numlayers": numlayers,
        "layer_top": layer_top,
        "layer_bottom": layer_bottom,
    }
