import numpy as np 
import torch
import kaolin as kal
def sdSphere(p):
    SPHERERAD = 0.5
    return np.linalg.norm(p)-SPHERERAD

def sdLink(p):
    #parameters
    le = 0.2
    r1 = 0.21
    r2 = 0.1
    q = np.array([p[0], max(abs(p[1])-le,0.0), p[2]])
    return np.linalg.norm(np.array([np.linalg.norm(q[0:2])-r1, q[2]])) - r2

def sdBox(p):
    SDBOXSIZE  = [1,1,1]
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return  np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0],max(q[1],q[2])),0.0)

def example_unit_cube_object(num_points=100000, yms=1e5, prs=0.45, rhos=100, DEVICE='cuda' ,DTYPE=torch.float):
    uniform_points = np.random.uniform([-0.5,-0.5,-0.5], [0.5,0.5,0.5], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdBox, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]
    return torch.tensor(X0, device=DEVICE, dtype=DTYPE), torch.tensor(X0_sdfval, device=DEVICE, dtype=DTYPE), torch.tensor(YMs, device=DEVICE, dtype=DTYPE), torch.tensor(PRs, device=DEVICE, dtype=DTYPE), torch.tensor(Rhos, device=DEVICE, dtype=DTYPE), torch.tensor(appx_vol, device=DEVICE, dtype=DTYPE)

def example_unit_cube_object_mesh(resolution=20, yms=1e5, prs=0.45, rhos=100, approx_volume=1.0, DEVICE='cuda' ,DTYPE=torch.float):

    cube_mesh = create_unit_cube_mesh(resolution=resolution,DEVICE=DEVICE) 

    cube_mesh.vertices = kal.ops.pointcloud.center_points(cube_mesh.vertices.unsqueeze(0), normalize=True).squeeze(0) 
    pts = cube_mesh.vertices
    yms = torch.full((pts.shape[0],), yms, device=DEVICE)
    prs = torch.full((pts.shape[0],), prs, device=DEVICE)
    rhos = torch.full((pts.shape[0],), rhos, device=DEVICE)
    approx_volume = torch.tensor([approx_volume], dtype=torch.float32, device=DEVICE)

    return cube_mesh, pts, yms, prs, rhos, approx_volume

def create_unit_cube_mesh(resolution=10, DEVICE='cuda'):
    """
    Create a unit cube mesh with specified resolution
    
    Args:
        resolution: Number of subdivisions along each edge (higher = more detailed mesh)
        device: torch device ('cuda' or 'cpu')
    
    Returns:
        mesh: Object with .vertices and .faces attributes (similar to kaolin mesh format)
    """
    
    # Create vertices for a subdivided cube
    # Generate grid points from -0.5 to 0.5 (unit cube centered at origin)
    x = np.linspace(-0.5, 0.5, resolution + 1)
    y = np.linspace(-0.5, 0.5, resolution + 1)
    z = np.linspace(-0.5, 0.5, resolution + 1)
    
    vertices = []
    faces = []
    
    # Function to get vertex index in the flattened array
    def get_vertex_index(i, j, k, res):
        return i * (res + 1) * (res + 1) + j * (res + 1) + k
    
    # Generate all vertices (including interior ones for a solid cube)
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            for k in range(resolution + 1):
                vertices.append([x[i], y[j], z[k]])
    
    # Generate faces for the cube (6 faces, each subdivided)
    res = resolution
    
    # Front face (z = 0.5)
    for i in range(res):
        for j in range(res):
            v1 = get_vertex_index(i, j, res, res)
            v2 = get_vertex_index(i + 1, j, res, res)
            v3 = get_vertex_index(i + 1, j + 1, res, res)
            v4 = get_vertex_index(i, j + 1, res, res)
            
            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Back face (z = -0.5)
    for i in range(res):
        for j in range(res):
            v1 = get_vertex_index(i, j, 0, res)
            v2 = get_vertex_index(i, j + 1, 0, res)
            v3 = get_vertex_index(i + 1, j + 1, 0, res)
            v4 = get_vertex_index(i + 1, j, 0, res)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Right face (x = 0.5)
    for j in range(res):
        for k in range(res):
            v1 = get_vertex_index(res, j, k, res)
            v2 = get_vertex_index(res, j + 1, k, res)
            v3 = get_vertex_index(res, j + 1, k + 1, res)
            v4 = get_vertex_index(res, j, k + 1, res)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Left face (x = -0.5)
    for j in range(res):
        for k in range(res):
            v1 = get_vertex_index(0, j, k, res)
            v2 = get_vertex_index(0, j, k + 1, res)
            v3 = get_vertex_index(0, j + 1, k + 1, res)
            v4 = get_vertex_index(0, j + 1, k, res)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Top face (y = 0.5)
    for i in range(res):
        for k in range(res):
            v1 = get_vertex_index(i, res, k, res)
            v2 = get_vertex_index(i, res, k + 1, res)
            v3 = get_vertex_index(i + 1, res, k + 1, res)
            v4 = get_vertex_index(i + 1, res, k, res)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Bottom face (y = -0.5)
    for i in range(res):
        for k in range(res):
            v1 = get_vertex_index(i, 0, k, res)
            v2 = get_vertex_index(i + 1, 0, k, res)
            v3 = get_vertex_index(i + 1, 0, k + 1, res)
            v4 = get_vertex_index(i, 0, k + 1, res)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Convert to torch tensors
    vertices = torch.tensor(vertices, device=DEVICE, dtype=torch.float32)
    faces = torch.tensor(faces, device=DEVICE, dtype=torch.long)
    
    # Create a simple mesh object (similar to kaolin mesh structure)
    class SimpleMesh:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces
    
    return SimpleMesh(vertices, faces)


def example_unit_sphere_object(num_points=100000, yms=1e6, prs=0.45, rhos=1000, DEVICE='cuda' ,DTYPE=torch.float):
    uniform_points = np.random.uniform([-3,-3,-3], [3,3,3], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdSphere, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]

    return torch.tensor(X0, device=DEVICE, dtype=DTYPE), torch.tensor(X0_sdfval, device=DEVICE, dtype=DTYPE), torch.tensor(YMs, device=DEVICE, dtype=DTYPE), torch.tensor(PRs, device=DEVICE, dtype=DTYPE), torch.tensor(Rhos, device=DEVICE, dtype=DTYPE), torch.tensor(appx_vol, device=DEVICE, dtype=DTYPE)
