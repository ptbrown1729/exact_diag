import ed_geometry as gm
import ed_symmetry as sym
import ed_fermions as hub

geom = gm.Geometry.createSquareGeometry(2, 2, 0, 0, 0, 0)
h = hub.fermions(geom, 0, 1, 1)
ham = h.createH()

