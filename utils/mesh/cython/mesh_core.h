#ifndef MESH_CORE_HPP_
#define MESH_CORE_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

class point
{
public:
    float x;
    float y;

    float dot(point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    point operator-(const point &p)
    {
        point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    point operator+(const point &p)
    {
        point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    point operator*(float s)
    {
        point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};

bool isPointInTri(point p, point p0, point p1, point p2, int h, int w);
void get_point_weight(float *weight, point p, point p0, point p1, point p2);

void _render_colors_core(
    float *image, float *vertices, int *triangles,
    float *colors,
    float *depth_buffer,
    float *raw_image,
    float *raw_depth_buffer,
    int nver, int ntri,
    int h, int w, int c);

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

void _render_texture_core(float *image, float *vertices, int *triangles,
                          float *texture, float *tex_coords, int *tex_triangles,
                          float *depth_buffer,
                          int nver, int tex_nver, int ntri,
                          int h, int w, int c,
                          int tex_h, int tex_w, int tex_c,
                          int mapping_type);

#endif