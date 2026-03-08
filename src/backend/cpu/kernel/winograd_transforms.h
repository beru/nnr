#pragma once
// Scalar Winograd F(4x4, 3x3) transform functions.
// Shared between platform-neutral and SIMD implementations.

namespace nnr {

// Pre-transform one 3x3 filter to 6x6 Winograd domain: U = G g GT
inline void winograd_filter_transform(float u[36], const float g[9]) {
    float tmp[18];
    for (int j = 0; j < 3; ++j) {
        float g0 = g[0*3+j], g1 = g[1*3+j], g2 = g[2*3+j];
        tmp[0*3+j] =  g0 * (1.0f/4);
        tmp[1*3+j] = (g0 + g1 + g2) * (-1.0f/6);
        tmp[2*3+j] = (g0 - g1 + g2) * (-1.0f/6);
        tmp[3*3+j] =  g0 * (1.0f/24) + g1 * (1.0f/12) + g2 * (1.0f/6);
        tmp[4*3+j] =  g0 * (1.0f/24) - g1 * (1.0f/12) + g2 * (1.0f/6);
        tmp[5*3+j] =  g2;
    }
    for (int i = 0; i < 6; ++i) {
        float t0 = tmp[i*3+0], t1 = tmp[i*3+1], t2 = tmp[i*3+2];
        u[i*6+0] =  t0 * (1.0f/4);
        u[i*6+1] = (t0 + t1 + t2) * (-1.0f/6);
        u[i*6+2] = (t0 - t1 + t2) * (-1.0f/6);
        u[i*6+3] =  t0 * (1.0f/24) + t1 * (1.0f/12) + t2 * (1.0f/6);
        u[i*6+4] =  t0 * (1.0f/24) - t1 * (1.0f/12) + t2 * (1.0f/6);
        u[i*6+5] =  t2;
    }
}

// Transform one 6x6 input tile to Winograd domain: V = BT d B
// BT (consistent with G and AT for F(4x4, 3x3)):
//   4  0 -5  0  1  0
//   0 -4 -4  1  1  0
//   0  4 -4 -1  1  0
//   0 -2 -1  2  1  0
//   0  2 -1 -2  1  0
//   0  4  0 -5  0  1
inline void winograd_input_transform(float v[36], const float d[36]) {
    float tmp[36];
    for (int j = 0; j < 6; ++j) {
        float d0 = d[0*6+j], d1 = d[1*6+j], d2 = d[2*6+j];
        float d3 = d[3*6+j], d4 = d[4*6+j], d5 = d[5*6+j];
        tmp[0*6+j] =  4*d0       - 5*d2        + d4;
        tmp[1*6+j] =      - 4*d1 - 4*d2 +   d3 + d4;
        tmp[2*6+j] =        4*d1 - 4*d2 -   d3 + d4;
        tmp[3*6+j] =      - 2*d1 -   d2 + 2*d3 + d4;
        tmp[4*6+j] =        2*d1 -   d2 - 2*d3 + d4;
        tmp[5*6+j] =        4*d1       - 5*d3        + d5;
    }
    for (int i = 0; i < 6; ++i) {
        float t0 = tmp[i*6+0], t1 = tmp[i*6+1], t2 = tmp[i*6+2];
        float t3 = tmp[i*6+3], t4 = tmp[i*6+4], t5 = tmp[i*6+5];
        v[i*6+0] =  4*t0       - 5*t2        + t4;
        v[i*6+1] =      - 4*t1 - 4*t2 +   t3 + t4;
        v[i*6+2] =        4*t1 - 4*t2 -   t3 + t4;
        v[i*6+3] =      - 2*t1 -   t2 + 2*t3 + t4;
        v[i*6+4] =        2*t1 -   t2 - 2*t3 + t4;
        v[i*6+5] =        4*t1       - 5*t3        + t5;
    }
}

// Transform 6x6 Winograd product back to 4x4 output tile: Y = AT M A
// AT (4x6):
//   1  1  1  1  1  0
//   0  1 -1  2 -2  0
//   0  1  1  4  4  0
//   0  1 -1  8 -8  1
inline void winograd_output_transform(float y[16], const float m[36]) {
    float tmp[24];
    for (int j = 0; j < 6; ++j) {
        float m0 = m[0*6+j], m1 = m[1*6+j], m2 = m[2*6+j];
        float m3 = m[3*6+j], m4 = m[4*6+j], m5 = m[5*6+j];
        tmp[0*6+j] = m0 + m1 + m2 +   m3 +   m4;
        tmp[1*6+j] =      m1 - m2 + 2*m3 - 2*m4;
        tmp[2*6+j] =      m1 + m2 + 4*m3 + 4*m4;
        tmp[3*6+j] =      m1 - m2 + 8*m3 - 8*m4 + m5;
    }
    for (int i = 0; i < 4; ++i) {
        float t0 = tmp[i*6+0], t1 = tmp[i*6+1], t2 = tmp[i*6+2];
        float t3 = tmp[i*6+3], t4 = tmp[i*6+4], t5 = tmp[i*6+5];
        y[i*4+0] = t0 + t1 + t2 +   t3 +   t4;
        y[i*4+1] =      t1 - t2 + 2*t3 - 2*t4;
        y[i*4+2] =      t1 + t2 + 4*t3 + 4*t4;
        y[i*4+3] =      t1 - t2 + 8*t3 - 8*t4 + t5;
    }
}

} // namespace nnr
