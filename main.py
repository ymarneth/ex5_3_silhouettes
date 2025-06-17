import numpy as np
import cv2
from skimage.measure import marching_cubes
import vedo
import trimesh


def project_voxels(voxels_coords, angle_deg, image_size):
    theta = np.radians(angle_deg)
    # Camera on circle around Y axis, rotation about Y axis by angle
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotated = R @ voxels_coords

    # Orthographic projection: drop z coordinate, keep x and y
    proj_x = rotated[0, :]
    proj_y = rotated[1, :]

    # Convert normalized coords [-1,1] to image pixels
    px = ((proj_x + 1) / 2 * (image_size[1] - 1)).astype(int)
    py = ((1 - (proj_y + 1) / 2) * (image_size[0] - 1)).astype(int)  # y inverted for image coords

    # Clip to valid range
    px = np.clip(px, 0, image_size[1] - 1)
    py = np.clip(py, 0, image_size[0] - 1)

    return px, py


def accumulate_votes(votes, angle_deg, silhouette_mask, voxel_coords):
    px, py = project_voxels(voxel_coords, angle_deg, silhouette_mask.shape)
    inside = silhouette_mask[py, px] > 0
    votes_flat = votes.ravel()
    votes_flat += inside.astype(np.uint8)
    return votes_flat.reshape(votes.shape)


def main():
    voxel_grid_size = 256
    image_path_template = '/workspace/data/slicesHuman/out{n:03d}.png'
    output_path = '/workspace/human_mesh.obj'

    step_size = 2
    angles = np.arange(0, 360, step_size)

    # Initialize voxel grid
    votes = np.zeros((voxel_grid_size, voxel_grid_size, voxel_grid_size), dtype=np.uint8)

    # Define normalized voxel coordinates [-1, 1]
    x = np.linspace(-1, 1, voxel_grid_size)
    y = np.linspace(-1, 1, voxel_grid_size)
    z = np.linspace(-1, 1, voxel_grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]) 

    # Load one image to get size
    sample_img = cv2.imread(image_path_template.format(n=0), cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise FileNotFoundError("Could not read sample image at angle 0")

    # Voting pass
    for angle in angles:
        print(f'Processing angle: {angle}°')
        mask = cv2.imread(image_path_template.format(n=angle), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        votes = accumulate_votes(votes, angle, mask, voxel_coords)

    # Threshold to keep voxels that appear in x% of views
    threshold = int(len(angles) * 0.95)
    voxels = votes >= threshold

    if np.sum(voxels) == 0:
        raise ValueError("No voxels remain after thresholding. Try reducing the threshold.")

    verts, faces, normals, values = marching_cubes(voxels.astype(float), level=0.5)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    vedo.Mesh([mesh.vertices, mesh.faces]).show(title="Mesh from Silhouettes")
    mesh.export(output_path, file_type='obj')


if __name__ == "__main__":
    main()
