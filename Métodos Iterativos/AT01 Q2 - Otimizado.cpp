#include <iostream>
#include <cmath>
#include <stdfloat> // Para std::float16_t no C++23

// Polinômio f(x) = x^3 + x - 4 em Template para aceitar qualquer precisão
template <typename T>
CUDA_CALLABLE inline T f_eval(T x) {
    return x * x * x + x - static_cast<T>(4.0);
}

// Extrai direção -1 ou +1 através do bit de sinal em tempo O(1)
template <typename T>
inline int get_direction(T x) {
    T fx = f_eval(x);
    return (fx < static_cast<T>(0.0)) - (fx > static_cast<T>(0.0));
}

double bissecção_precisao_mista(double a0, double b0, int total_iter) {
    double x = a0 + (b0 - a0) / 2.0;
    double step = (b0 - a0) / 2.0;

    for (int i = 2; i <= total_iter; ++i) {
        step /= 2.0;
        int d_i = 0;

        // Limiares para troca dinâmica de precisão no hardware
        double current_fx = std::abs(f_eval(x));

        if (current_fx > 1e-3) {
            // Estágio 1: Execução ultra-rápida em fp16 (half)
            auto x_half = static_cast<std::float16_t>(x);
            d_i = get_direction(x_half);
        } 
        else if (current_fx > 1e-7) {
            // Estágio 2: Execução em fp32 (float)
            auto x_float = static_cast<float>(x);
            d_i = get_direction(x_float);
        } 
        else {
            // Estágio 3: Resolução máxima em fp64 (double)
            d_i = get_direction(x);
        }

        x += d_i * step; // Atualização do estado do acumulador
    }

    return x;
}