#ifndef SIMPLEX_NOISE_H
#define SIMPLEX_NOISE_H

#include <cmath>
#include <ctime>
#include <vector>
#include <array>

// Структура для представления градиентов
struct Grad {
    float x, y, z;
    Grad(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Grad() : x(0.0f), y(0.0f), z(0.0f) {}
    float dot2(float x_, float y_) const { return x * x_ + y * y_; }
    float dot3(float x_, float y_, float z_) const { return x * x_ + y * y_ + z * z_; }
};

// Класс для генерации шумов по алгоритму Simplex
class SimplexNoise {
public:
    SimplexNoise(unsigned int seed = std::time(nullptr)) {
        grad3 = {
            Grad(1,1,0), Grad(-1,1,0), Grad(1,-1,0), Grad(-1,-1,0),
            Grad(1,0,1), Grad(-1,0,1), Grad(1,0,-1), Grad(-1,0,-1),
            Grad(0,1,1), Grad(0,-1,1), Grad(0,1,-1), Grad(0,-1,-1)
        };
        p = {
            151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
            129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
            49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        };
        perm.resize(512);
        gradP.resize(512);
        seed_noise(seed);
    }

    void seed_noise(unsigned int seed) {
        if (seed < 256) seed |= seed << 8;
        for (int i = 0; i < 256; ++i) {
            int v = (i & 1) ? (p[i] ^ (seed & 255)) : (p[i] ^ ((seed >> 8) & 255));
            perm[i] = perm[i + 256] = v;
            gradP[i] = gradP[i + 256] = grad3[v % 12];
        }
    }

    float d2(float xin, float yin) const {
        static const float F2 = 0.5f * (std::sqrt(3.0f) - 1.0f);
        static const float G2 = (3.0f - std::sqrt(3.0f)) / 6.0f;

        float s = (xin + yin) * F2;
        int i = std::floor(xin + s);
        int j = std::floor(yin + s);
        float t = (i + j) * G2;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = xin - X0;
        float y0 = yin - Y0;

        int i1, j1;
        if (x0 > y0) { i1 = 1; j1 = 0; }
        else { i1 = 0; j1 = 1; }

        float x1 = x0 - i1 + G2;
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2;
        float y2 = y0 - 1.0f + 2.0f * G2;

        int ii = i & 255;
        int jj = j & 255;
        const Grad& gi0 = gradP[ii + perm[jj]];
        const Grad& gi1 = gradP[ii + i1 + perm[jj + j1]];
        const Grad& gi2 = gradP[ii + 1 + perm[jj + 1]];

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        float n0 = (t0 < 0) ? 0.0f : std::pow(t0, 4.0f) * gi0.dot2(x0, y0);

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        float n1 = (t1 < 0) ? 0.0f : std::pow(t1, 4.0f) * gi1.dot2(x1, y1);

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        float n2 = (t2 < 0) ? 0.0f : std::pow(t2, 4.0f) * gi2.dot2(x2, y2);

        return 70.0f * (n0 + n1 + n2);
    }

    float d1(float x) const {
        return d2(x * 1.2f, -x * 0.7f);
    }

private:
    std::vector<Grad> grad3;
    std::vector<int> perm;
    std::vector<Grad> gradP;
    std::array<int, 256> p;
};

// Глобальный объект для использования в других частях программы
extern SimplexNoise simplex_noise;

#endif // SIMPLEX_NOISE_H
