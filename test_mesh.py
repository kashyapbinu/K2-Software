import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

occ = gmsh.model.occ

# Box
box = occ.addBox(-2, -2, -2, 4, 4, 4)
# Cylinder hole
cyl = occ.addCylinder(0, 0, -3, 0, 0, 6, 0.5)

fluid, _ = occ.cut([(3, box)], [(3, cyl)])
occ.synchronize()

# Find cylinder wall
surfs = gmsh.model.getEntities(2)
wall_surfs = []
for dim, tag in surfs:
    com = occ.getCenterOfMass(dim, tag)
    if abs(com[0]**2 + com[1]**2 - 0.25) < 0.1:
        wall_surfs.append(tag)

print("Wall surfs:", wall_surfs)

gmsh.model.mesh.generate(2)

# Try geo.extrudeBoundaryLayer
dimtags = [(2, s) for s in wall_surfs]

# Option 1: Reverse normals before extrusion
gmsh.model.mesh.reverse(dimtags)

out = gmsh.model.geo.extrudeBoundaryLayer(
    dimtags,
    numElements=[2, 2],
    heights=[0.05, 0.1],
    recombine=True
)
gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)
gmsh.write("test_mesh.msh")
gmsh.finalize()
