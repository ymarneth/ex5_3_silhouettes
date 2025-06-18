import numpy as np
import cv2
from skimage.measure import marching_cubes
import vedo
import trimesh
import os
import imageio


DATA_PATH = "/workspace/data/slicesHuman"
IMAGE_TEMPLATE = "out{n:03d}.png"
OUTPUT_PATH = "/workspace/output"
OBJ_OUTPUT_FILENAME = "human_mesh.obj"
GIF_OUTPUT_FILENAME = "human.gif"

VOXEL_GRID_SIZE = 256   # Size of the voxel grid
STEP_SIZE = 1           # Step size for angles in degrees (assuming images are taken every 1 degrees), can be used to make rougher estimations for debugging
THRESHOLD = 0.99        # Percentage of views a voxel must appear in to be included


def load_images(image_path: str, angles: np.ndarray) -> list:
    images = []
    for angle in angles:
        img = cv2.imread(image_path.format(n=angle), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path.format(n=angle)}")
        images.append({'angle': angle, 'img': img})
    return images


# Generate a 3D grid of normalized voxel coordinates in the range [-1, 1]
def generate_voxel_coords(size: int) -> np.ndarray:
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    return np.vstack([X.ravel(), Y.ravel(), Z.ravel()])


def carve_voxels(
    images: list, 
    voxel_coords: np.ndarray,
    grid_size: int,
    threshhold: float = 0.95
) -> np.ndarray:
    votes = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint16)
    
    # Loop through silhouette images and accumulate voxel votes
    # Voxel votes are accumulated based on whether the voxel is inside the silhouette in each view
    for image in images:
        angle = image['angle']
        img = image['img']
        print(f'Processing angle: {angle}°')
        mask = (img > 127).astype(np.uint8) # binarize
        votes = accumulate_votes(votes, angle, mask, voxel_coords)

    # Threshold to keep voxels that appear in x% of views
    threshold = int(len(images) * threshhold)
    voxels = votes >= threshold

    if np.sum(voxels) == 0:
        raise ValueError("No voxels remain after thresholding. Try reducing the threshold.")
    
    return voxels


def accumulate_votes(
    votes: np.ndarray,
    angle_deg: float,
    silhouette_mask: np.ndarray,
    voxel_coords: np.ndarray
) -> np.ndarray:
    # project voxel coordinates to image coordinates
    px, py = project_voxels(voxel_coords, angle_deg, silhouette_mask.shape)
    isInside = silhouette_mask[py, px] > 0
    
    # Accumulate votes for voxels inside the silhouette
    votes_flat = votes.ravel()
    votes_flat += isInside.astype(np.uint8)
    
    return votes_flat.reshape(votes.shape)


def project_voxels(
    voxels_coords: np.ndarray, 
    angle_deg: float, 
    image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
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
    px = ((proj_x + 1) / 2 * (image_size[1] - 1)).astype(np.int32)
    py = ((1 - (proj_y + 1) / 2) * (image_size[0] - 1)).astype(np.int32)  # y inverted for image coords

    # Clip to valid range
    px = np.clip(px, 0, image_size[1] - 1)
    py = np.clip(py, 0, image_size[0] - 1)

    return px, py


def create_mesh(voxels: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _, _ = marching_cubes(voxels.astype(float), level=0.5)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def save_mesh(mesh: trimesh.Trimesh, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mesh.export(output_path, file_type='obj')
    print(f"Mesh saved to {output_path}")


def generate_gif(images: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = []
    for image in images:
        img = image['img']
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))  # convert to rgb for gif

    imageio.mimsave(output_path, frames, duration=0.1)
    print(f"GIF saved to {output_path}")


def main():
    angles = np.arange(0, 360, STEP_SIZE)
    image_path = f'{DATA_PATH}/{IMAGE_TEMPLATE}'
    
    # preload images to make the algorithm faster
    images = load_images(image_path, angles)
    
    # generate for debugging
    generate_gif(images, os.path.join(OUTPUT_PATH, GIF_OUTPUT_FILENAME))
    
    voxel_coords = generate_voxel_coords(VOXEL_GRID_SIZE)
    voxels = carve_voxels(images, voxel_coords, VOXEL_GRID_SIZE, THRESHOLD)
    mesh = create_mesh(voxels)

    vedo.Mesh([mesh.vertices, mesh.faces]).show(title="Mesh from Silhouettes")
    
    save_mesh(mesh, os.path.join(OUTPUT_PATH, OBJ_OUTPUT_FILENAME))


if __name__ == "__main__":
    main()
