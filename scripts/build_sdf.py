import argparse
from mrrt.sdf import SDFMesh


def main():
    parser = argparse.ArgumentParser(description="Compute SDF npz for a mesh")
    parser.add_argument("mesh", help="Path to mesh file")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="Voxel size for the SDF grid")
    parser.add_argument("--padding", type=float, default=0.1,
                        help="Padding to add around the mesh bounding box")
    parser.add_argument("--export_mesh", type=str, default=None,
                        help="Optional path to export an obj via marching cubes")
    args = parser.parse_args()

    m = SDFMesh(args.mesh)
    m.fit(voxel_size=args.voxel_size, padding=args.padding)
    if args.export_mesh:
        m.to_mesh(args.export_mesh)


if __name__ == "__main__":
    main()
