#ifndef __MATH3D_H__
#define __MATH3D_H__


#include <assert.h>
#include <xmmintrin.h>

// Can't use SSE 4.1 on clang llvm compiler for some reason :C
// Maybe will research more indepth later on why it isn't enabled by default
// Other library I found was accelerator.h, but can't use that for my projects
// Ended up using SSE 2


// It's better to use FPU (single instruction, single data) for vec3
// because extra wasted 32 bits or more (need to be aligned to 16)
// can cause cache miss. But, for science, right?
class vec3 {
public:
    vec3();
    vec3(__m128 v);
    vec3(float x, float y, float z);
    vec3 operator+(const vec3 &) const;
    vec3 operator-(const vec3 &) const;
    vec3 operator-() const;
    
    vec3 operator*(float) const;
    float dot(const vec3 &) const;
    vec3 cross(const vec3 &) const;
    
    void scale(float);
    void scale(float, float, float);
    float* ptr();

    float length() const;
    vec3 normalize() const;
    float angle(const vec3 &) const;
    void print();

    __m128 v;
    
} __attribute__((aligned(16)));


class vec4 {
public:
    vec4();
    vec4(__m128 v);
    vec4(float x, float y, float z, float w);
    vec4 operator+(const vec4&) const;
    vec4 operator-(const vec4&) const;
    vec4 operator-() const;
    
    vec4 operator*(const vec4&) const;
    vec4 operator*(const float&) const;
    void dehomogenize();
    
    float* ptr();
    void print();
    
    __m128 v;
    
} __attribute__((aligned(16)));


class mat4 {
public:
    mat4();
    mat4(__m128[4]);
    mat4(const float (&el)[4][4]);
    mat4(float, float, float, float,
         float, float, float, float,
         float, float, float, float,
         float, float, float, float);
    
    void set(float, float, float, float,
             float, float, float, float,
             float, float, float, float,
             float, float, float, float);
    
    mat4 operator*(const mat4&) const;
    vec4 operator*(const vec4&) const;
    mat4& operator=(const mat4&);
    
    void makeIdentity();
    void makeTranspose();
    
    float* ptr();
    void print();
    
    __m128 m[4]; // row-major
} __attribute__((aligned(16)));

#endif
