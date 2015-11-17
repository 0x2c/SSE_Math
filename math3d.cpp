#include "math3d.h"
#include <iostream>
#include <math.h>



/**
 *                                      Vector3
 * ======================================================================================
 */

vec3::vec3() : v(_mm_setzero_ps()) {}

vec3::vec3(__m128 m) : v(m) {}

vec3::vec3(float x, float y, float z) { v = _mm_set_ps(0.0f, z, y, x); }

vec3 vec3::operator+(const vec3 &b) const { return _mm_add_ps(v, b.v); }

vec3 vec3::operator-(const vec3 &b) const { return _mm_sub_ps(v, b.v); }

vec3 vec3::operator-() const { return _mm_sub_ps(_mm_set1_ps(0.0f), v); }

vec3 vec3::operator*(float f) const { return vec3(_mm_mul_ps(_mm_set1_ps(f), v)); }

float vec3::dot(const vec3 &b) const {
    // make sure w component is 0
    __m128 temp = _mm_mul_ps(v, b.v);
    __m128 temp2 = _mm_shuffle_ps(temp, temp, 0xFE);
    temp2 = _mm_add_ps(temp, temp2);
    return _mm_cvtss_f32(_mm_add_ps(temp2, _mm_shuffle_ps(temp, temp, 0xFD)));
}

vec3 vec3::cross(const vec3 &b) const {
    const static uint32_t mask0 = _MM_SHUFFLE( 0, 0, 2, 1 );
    const static uint32_t mask1 = _MM_SHUFFLE( 0, 1, 0, 2 );
    return _mm_sub_ps(
        _mm_mul_ps(_mm_shuffle_ps(v, v, mask0), _mm_shuffle_ps(b.v, b.v, mask1)),
        _mm_mul_ps(_mm_shuffle_ps(v, v, mask1), _mm_shuffle_ps(b.v, b.v, mask0))
    );
}

void vec3::scale(float f) {
    v = _mm_mul_ps(_mm_set1_ps(f), v);
}

void vec3::scale(float f1, float f2, float f3) {
    v = _mm_mul_ps(_mm_set_ps(0.0f, f3, f2, f1),  v);
}

float* vec3::ptr() {
    return reinterpret_cast<float *>(&v);
}

float vec3::length() const {
    __m128 temp = _mm_mul_ps(v, v);
    __m128 temp2 = _mm_shuffle_ps(temp, temp, 0xFD);
    temp2 = _mm_add_ps(temp, temp2);
    temp2 = _mm_add_ps(temp2, _mm_shuffle_ps(temp, temp, 0xFE));
    return _mm_cvtss_f32(_mm_sqrt_ps(temp2));
}

vec3 vec3::normalize() const {
    __m128 temp = _mm_mul_ps(v, v);
    __m128 temp2 = _mm_shuffle_ps(temp, temp, 0xFD);
    temp2 = _mm_add_ps(temp, temp2);
    temp2 = _mm_add_ps(temp2, _mm_shuffle_ps(temp, temp, 0xFE));
    __m128 sqr = _mm_rsqrt_ps(temp2);
    return _mm_mul_ps(v, _mm_shuffle_ps(sqr, sqr, 0x00) );
}

float vec3::angle(const vec3 &b) const {
    float l = b.length();
    return acosf( dot(b)/l ) / l;
}

void vec3::print() {
    printf("vec3 : %g %g %g\n", v[0], v[1], v[2]);
}



/**
 *                                      Vector4
 * ======================================================================================
 */

vec4::vec4() : v(_mm_setzero_ps()) {}

vec4::vec4(__m128 m) : v(m) {}

vec4::vec4(float x, float y, float z, float w) { v = _mm_set_ps(w, z, y, x); }

vec4 vec4::operator+(const vec4 &b) const { return _mm_add_ps(v, b.v); }

vec4 vec4::operator-(const vec4 &b) const { return _mm_sub_ps(v, b.v); }

vec4 vec4::operator-() const { return _mm_sub_ps(_mm_set1_ps(0.0f), v); }

vec4 vec4::operator*(const vec4 &b) const { return _mm_mul_ps(v, b.v); }

vec4 vec4::operator*(const float &f) const { return _mm_mul_ps(_mm_set1_ps(f), v); }

void vec4::dehomogenize() {
    v = _mm_div_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)));
}

float* vec4::ptr() {
    return reinterpret_cast<float *>(&v);
}

void vec4::print() {
    printf("vec4 : %g %g %g %g\n", v[0], v[1], v[2], v[3]);
}

/**
 *                                      Matrix4
 * ======================================================================================
 */

mat4::mat4() {
    m[0] = _mm_setzero_ps();
    m[1] = _mm_setzero_ps();
    m[2] = _mm_setzero_ps();
    m[3] = _mm_setzero_ps();
}

mat4::mat4(__m128 el[4]) {
    m[0] = el[0];
    m[1] = el[1];
    m[2] = el[2];
    m[3] = el[3];
}

mat4::mat4(const float (&el)[4][4]) {
    m[0] = _mm_set_ps(el[0][3], el[0][2], el[0][1], el[0][0]);
    m[1] = _mm_set_ps(el[1][3], el[1][2], el[1][1], el[1][0]);
    m[2] = _mm_set_ps(el[2][3], el[2][2], el[2][1], el[2][0]);
    m[3] = _mm_set_ps(el[3][3], el[3][2], el[3][1], el[3][0]);
}

mat4::mat4(float m00, float m01, float m02, float m03,
           float m10, float m11, float m12, float m13,
           float m20, float m21, float m22, float m23,
           float m30, float m31, float m32, float m33) {
    m[0] = _mm_set_ps(m03, m02, m01, m00);
    m[1] = _mm_set_ps(m13, m12, m11, m10);
    m[2] = _mm_set_ps(m23, m22, m21, m20);
    m[3] = _mm_set_ps(m33, m32, m31, m30);
    
}

void mat4::set(float m00, float m01, float m02, float m03,
               float m10, float m11, float m12, float m13,
               float m20, float m21, float m22, float m23,
               float m30, float m31, float m32, float m33) {
    m[0] = _mm_set_ps(m03, m02, m01, m00);
    m[1] = _mm_set_ps(m13, m12, m11, m10);
    m[2] = _mm_set_ps(m23, m22, m21, m20);
    m[3] = _mm_set_ps(m33, m32, m31, m30);
}

// It's 20% faster but wasn't the 2x speed up I was hoping for.
// Will have to investigate later.
mat4 mat4::operator*(const mat4& b) const {
    mat4 temp = b;
    temp.makeTranspose();
    __m128 r[4];
    for(int i = 0; i < 4; ++i) {
        __m128 c0 = _mm_mul_ps(m[i], temp.m[0]);
        __m128 c1 = _mm_mul_ps(m[i], temp.m[1]);
        __m128 c2 = _mm_mul_ps(m[i], temp.m[2]);
        __m128 c3 = _mm_mul_ps(m[i], temp.m[3]);
        _MM_TRANSPOSE4_PS(c0, c1, c2, c3);
        r[i] = _mm_add_ps(c0, _mm_add_ps(c1, _mm_add_ps(c2, c3)));
    }
    return r;
}

vec4 mat4::operator*(const vec4& b) const {
    __m128 r0 = _mm_mul_ps(m[0], b.v);
    __m128 r1 = _mm_mul_ps(m[1], b.v);
    __m128 r2 = _mm_mul_ps(m[2], b.v);
    __m128 r3 = _mm_mul_ps(m[3], b.v);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    return _mm_add_ps(r0, _mm_add_ps(r1, _mm_add_ps(r2, r3)));
}

mat4& mat4::operator=(const mat4& b) {
    m[0] = b.m[0];
    m[1] = b.m[1];
    m[2] = b.m[2];
    m[3] = b.m[3];
    return *this;
}

void mat4::makeIdentity() {
   m[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
   m[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
   m[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
   m[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
}

void mat4::makeTranspose() {
    _MM_TRANSPOSE4_PS(m[0], m[1], m[2], m[3]);
}

float* mat4::ptr() {
    return reinterpret_cast<float *>(&m);
}

void mat4::print() {
    printf("mat4:: [%g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3] );
    printf("        %g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("        %g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("        %g %g %g %g]\n", m[3][0], m[3][1], m[3][2], m[3][3]);
    
}
