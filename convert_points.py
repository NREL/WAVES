from pathlib import Path

import pandas as pd
from pyproj import CRS
from pyproj.transformer import Transformer

def main(reference_point: tuple[float, float], file_name: Path) -> None:
    
    # Build the transformers and get the reference point
    crs = CRS.from_epsg(32718)
    proj = Transformer.from_crs(crs.geodetic_crs, crs)
    proj_reverse = Transformer.from_crs(crs, crs.geodetic_crs)
    
    base_easting, base_northing = proj.transform(*reference_point)
    
    # Load in the data
    df = pd.read_csv("base_floating_2022_layout.csv")

    spatial_coordinates = df[["easting", "northing"]].values + (base_easting + base_northing)
    coordinates = [proj_reverse.transform(e, n) for (e, n) in spatial_coordinates]

    df[["longitude", "latitude"]] = coordinates
    df.to_csv(file_name.with_stem(f"{file_name.stem}_converted_coordinates"), index=False)


if __name__ == "__main__":
    reference_point = (-124.708, 40.928)  # (lon, lat) or (x, y) in WGS-84
    file_name  = Path("./COE_layout_base.csv").resolve()
    
    main(reference_point, file_name)
