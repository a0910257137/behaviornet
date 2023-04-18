import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "mesh_core.h":

    void _render_colors_core(
        float* image, float* vertices, int* triangles, 
        float* colors, 
        float* depth_buffer,
        float *raw_image,
        float *raw_depth_buffer,
        int nver, int ntri,
        int h, int w, int c)

    void _render_mask_texture_core(
        float *image, 
        float *vertices,
        int *triangles,
        float *triangle_texture,
        float *triangle_depth,
        float *depth_buffer,
        float *raw_image,
        float *raw_depth_buffer,
        int nver, int ntri,
        int h, int w, int c);

    void _render_texture_core(
        float* image, float* vertices, int* triangles, 
        float* texture, float* tex_coords, int* tex_triangles, 
        float* depth_buffer,
        int nver, int tex_nver, int ntri, 
        int h, int w, int c, 
        int tex_h, int tex_w, int tex_c, 
        int mapping_type)



def render_colors_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] colors not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _render_colors_core(
        <float*> np.PyArray_DATA(image), 
        <float*> np.PyArray_DATA(vertices), 
        <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(colors), 
        <float*> np.PyArray_DATA(depth_buffer),
        <float*> np.PyArray_DATA(image),
        <float*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)

def render_mask_texture_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] triangle_texture not None, 
                np.ndarray[float, ndim=1, mode = "c"] triangle_depth not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _render_mask_texture_core(
        <float*> np.PyArray_DATA(image), 
        <float*> np.PyArray_DATA(vertices), 
        <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(triangle_texture), 
        <float*> np.PyArray_DATA(triangle_depth), 
        <float*> np.PyArray_DATA(depth_buffer),
        <float*> np.PyArray_DATA(image),
        <float*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)

def render_texture_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=3, mode = "c"] texture not None, 
                np.ndarray[float, ndim=2, mode = "c"] tex_coords not None, 
                np.ndarray[int, ndim=2, mode="c"] tex_triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int tex_nver, int ntri,
                int h, int w, int c,
                int tex_h, int tex_w, int tex_c,
                int mapping_type
                ):   
    _render_texture_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(texture), <float*> np.PyArray_DATA(tex_coords), <int*> np.PyArray_DATA(tex_triangles),  
        <float*> np.PyArray_DATA(depth_buffer),
        nver, tex_nver, ntri,
        h, w, c, 
        tex_h, tex_w, tex_c, 
        mapping_type)

