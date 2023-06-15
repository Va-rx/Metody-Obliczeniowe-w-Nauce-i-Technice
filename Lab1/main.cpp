//Wykonać obliczenia (dla zmiennych typu float, double, long double) wg podanych
//poniżej wzorów dla 101 równoodległych wartości x z przedziału [0.99, 1.01]

#include <iostream>
#include <cmath>
#include <limits>

float f1f(float);
float f2f(float);
float f3f(float);
float f4f(float);

double f1d(double);
double f2d(double);
double f3d(double);
double f4d(double);

long double f1ld(long double);
long double f2ld(long double);
long double f3ld(long double);
long double f4ld(long double);

int main() {
    // Typ nie ma znaczenia dla x
    double arr[101];
    double r = (1.01 - 0.99) / (101 - 1);
    for(int i = 0; i < 101; i++) {
        arr[i] = 0.99 + i*r;
    }
    

    float F1f[101];
    float F2f[101];
    float F3f[101];
    float F4f[101];

    double F1d[101];
    double F2d[101];
    double F3d[101];
    double F4d[101];

    long double F1ld[101];
    long double F2ld[101];
    long double F3ld[101];
    long double F4ld[101];


    for(int i = 0; i < 101; i++) {
        F1f[i] = f1f(arr[i]);
        F2f[i] = f2f(arr[i]);
        F3f[i] = f3f(arr[i]);
        F4f[i] = f4f(arr[i]);

        F1d[i] = f1d(arr[i]);
        F2d[i] = f2d(arr[i]);
        F3d[i] = f3d(arr[i]);
        F4d[i] = f4d(arr[i]);

        F1ld[i] = f1ld(arr[i]);
        F2ld[i] = f2ld(arr[i]);
        F3ld[i] = f3ld(arr[i]);
        F4ld[i] = f4ld(arr[i]);
    }
    return 0;
}

float f1f(float x) {
    return powf(x, 8) - 8*powf(x, 7) + 28*powf(x, 6) - 56*powf(x, 5) + 70*powf(x, 4)
    - 56*powf(x, 3) + 28*powf(x, 2) - 8*x + 1;
}

float f2f(float x) {
    return (((((((x-8)*x +28)*x - 56)*x + 70)*x - 56)*x + 28)*x - 8)*x +1;
}

float f3f(float x) {
    return powf(x-1, 8);
}

float f4f(float x) {
    return expf(8 * logf(std::abs(x-1)));
}

double f1d(double x) {
    return pow(x, 8) - 8*pow(x, 7) + 28*pow(x, 6) - 56*pow(x, 5) + 70*pow(x, 4)
           - 56*pow(x, 3) + 28*pow(x, 2) - 8*x + 1;
}

double f2d(double x) {
    return (((((((x-8)*x +28)*x - 56)*x + 70)*x - 56)*x + 28)*x - 8)*x +1;
}

double f3d(double x) {
    return pow(x-1, 8);
}

double f4d(double x) {
    return exp(8 * log(std::abs(x-1)));
}

long double f1ld(long double x) {
    return powl(x, 8) - 8*powl(x, 7) + 28*powl(x, 6) - 56*powl(x, 5) + 70*powl(x, 4)
           - 56*powl(x, 3) + 28*powl(x, 2) - 8*x + 1;
}

long double f2ld(long double x) {
    return (((((((x-8)*x +28)*x - 56)*x + 70)*x - 56)*x + 28)*x - 8)*x +1;
}

long double f3ld(long double x) {
    return powl(x-1, 8);
}

long double f4ld(long double x) {
    return expl(8 * logl(std::abs(x-1)));
}



