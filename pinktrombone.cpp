#include <iostream>
#include <numbers>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include "AL/al.h"
#include "AL/alc.h"
#include "simplex_noise.h"

class Glottis {
public:
    Glottis(int sample_rate)
        : sample_rate(sample_rate), time(0.0), phase(0.0), frequency(140.0), smooth_frequency(140.0),
          tenseness(1.0), loudness(1.0), vibrato_amount(0.0), vibrato_frequency(6.0), intensity(0.8), rd(3.0) {
        setup_waveform(0);
    }

    void setup_waveform(double lambda_val) {
        double frequency = this->frequency;
        double tenseness = this->tenseness;
        this->rd = 3 * (1 - tenseness);

        double ra = -0.01 + 0.048 * this->rd;
        double rk = 0.224 + 0.118 * this->rd;
        double rg = (rk / 4) * (0.5 + 1.2 * rk) / (0.11 * this->rd - ra * (0.5 + 1.2 * rk));

        this->ta = ra;
        this->tp = 1 / (2 * rg);
        this->te = this->tp + this->tp * rk;
        this->epsilon = 1 / this->ta;
        this->shift = std::exp(-this->epsilon * (1 - this->te));
        this->delta = 1 - this->shift;
        double rhs_integral = (1 / this->epsilon) * (this->shift - 1) + (1 - this->te) * this->shift;
        rhs_integral /= this->delta;
        double total_lower_integral = - (this->te - this->tp) / 2 + rhs_integral;

        this->omega = std::numbers::pi / this->tp;
        double s = std::sin(this->omega * this->te);
        double y = std::numbers::pi * s * total_lower_integral / (this->tp * 2);
        this->alpha = std::log(y) / (this->tp / 2 - this->te);
        this->e0 = -1 / (s * std::exp(this->alpha * this->te));
    }

    double normalized_waveform(double t) {
        if (t > this->te) {
            return (-std::exp(-this->epsilon * (t - this->te)) + this->shift) / this->delta;
        } else {
            return this->e0 * std::exp(this->alpha * t) * std::sin(this->omega * t);
        }
    }

    double noise_modulator() {
        double voiced = 0.1 + 0.2 * std::max(0.0, std::sin(2 * std::numbers::pi * this->phase * this->frequency));
        return this->tenseness * this->intensity * (voiced - 0.3) + 0.3;
    }

    double run_step(double lambda_val, double noise) {
        double dt = 1.0 / this->sample_rate;
        this->time += dt;
        this->phase += dt;

        if (this->phase > 1 / this->frequency) {
            this->phase -= 1 / this->frequency;
            setup_waveform(lambda_val);
        }

        double output = normalized_waveform(this->phase * this->frequency);
        double aspiration = (1 - std::sqrt(this->tenseness)) * noise_modulator() * noise * (0.2 + 0.02 * simplex_noise.d1(this->time * 1.99));
        return (output + aspiration) * this->loudness * this->intensity;
    }

    int sample_rate;
    double time, phase, frequency, smooth_frequency, tenseness, loudness;
    double vibrato_amount, vibrato_frequency, intensity, rd;
    double ta, tp, te, epsilon, shift, delta, omega, alpha, e0;
};

class Tract {
public:
    Tract(int sample_rate) : sample_rate(sample_rate), lip_reflection(-0.85), glottal_reflection(0.75), fade(0.999) {
        n = 44;
        R.resize(n);
        L.resize(n);
        junction_out_R.resize(n + 1);
        junction_out_L.resize(n + 1);
        reflection.resize(n + 1);
        new_reflection.resize(n + 1);
        diameter.resize(n, 1.5);
        area.resize(n, 1.5);
        lip_output = 0;
        nose_output = 0;
        calculate_reflections();
    }

    void calculate_reflections() {
        for (int i = 1; i < n; ++i) {
            if (area[i] + area[i - 1] == 0) {
                new_reflection[i] = 0.999;
            } else {
                new_reflection[i] = (area[i - 1] - area[i]) / (area[i - 1] + area[i]);
            }
        }
    }

    void run_step(double glottal_output, double turbulence_noise, double lambda_val) {
        junction_out_R[0] = L[0] * glottal_reflection + glottal_output;
        junction_out_L[n] = R[n - 1] * lip_reflection;

        for (int i = 1; i < n; ++i) {
            double r = (1 - lambda_val) * reflection[i] + lambda_val * new_reflection[i];
            double w = r * (R[i - 1] + L[i]);
            junction_out_R[i] = R[i - 1] - w;
            junction_out_L[i] = L[i] + w;
        }

        for (int i = 0; i < n; ++i) {
            R[i] = std::clamp(junction_out_R[i] * fade, -1.0, 1.0);
            L[i] = std::clamp(junction_out_L[i + 1] * fade, -1.0, 1.0);
        }

        lip_output = R[n - 1];
        reflection = new_reflection;
    }

    void finish_block() {
        calculate_reflections();
    }

    double lip_output;

private:
    int sample_rate;
    int n;
    double lip_reflection, glottal_reflection, fade;
    std::vector<double> R, L, junction_out_R, junction_out_L, reflection, new_reflection, diameter, area;
    double nose_output;
};

class PinkTromboneEngine {
public:
    Glottis glottis;
    Tract tract;

    PinkTromboneEngine(int sample_rate = 44100, int buffer_size = 2048)
        : sample_rate(sample_rate), buffer_size(buffer_size), glottis(sample_rate), tract(sample_rate), running(false) {}

    std::vector<float> generate_buffer() {
        std::vector<float> buffer(buffer_size, 0.0);
        for (int i = 0; i < buffer_size; ++i) {
            double lambda1 = i / double(buffer_size);
            double noise = rng();
            double glottal_output = glottis.run_step(lambda1, noise);
            tract.run_step(glottal_output, noise, lambda1);
            double lambda2 = (i + 0.5) / double(buffer_size);
            tract.run_step(glottal_output, noise, lambda2);
            buffer[i] = tract.lip_output * 0.125;
        }
        glottis.setup_waveform(1.0);
        tract.finish_block();
        return buffer;
    }

    void audio_loop() {
        ALCdevice* device = alcOpenDevice(NULL);
        ALCcontext* context = alcCreateContext(device, NULL);
        alcMakeContextCurrent(context);

        ALuint source;
        alGenSources(1, &source);

        running = true;
        while (running) {
            auto raw_data = generate_buffer();
            ALuint buffer;
            alGenBuffers(1, &buffer);
            alBufferData(buffer, AL_FORMAT_MONO16, raw_data.data(), raw_data.size() * sizeof(float), sample_rate);
            alSourcei(source, AL_BUFFER, buffer);
            alSourcePlay(source);

            while (true) {
                ALint state;
                alGetSourcei(source, AL_SOURCE_STATE, &state);
                if (state != AL_PLAYING) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            alDeleteBuffers(1, &buffer);
        }

        alDeleteSources(1, &source);
        alcDestroyContext(context);
        alcCloseDevice(device);
    }

    void start() {
        running = true;
        std::thread([this]() { audio_loop(); }).detach();
    }

    void stop() {
        running = false;
    }

private:
    int sample_rate, buffer_size;
    bool running;
    std::default_random_engine rng;
};

//int main() {
//    PinkTromboneEngine engine = PinkTromboneEngine(44100, 44100*5);
//
//    // Замер времени
//    auto start_time = std::chrono::high_resolution_clock::now();
//    engine.generate_buffer();
//
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration = end_time - start_time;
//
//    std::cout << "Время генерации звука: " << duration.count() << " секунд.\n";
//
//    engine.stop();  // Останавливаем генерацию звука
//    return 0;
//}

int main() {
    PinkTromboneEngine engine = PinkTromboneEngine(44100, 1024);
    engine.start();

    // Замер времени
    auto start_time = std::chrono::high_resolution_clock::now();

    // Паттерн для генерации частоты
    static float last_frequency = 340;
    while (true) {  // Бесконечный цикл
        auto elapsed_time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count();

        // Плавная синусоидальная модуляция с несколькими гармониками
        float frequency = 340 + 50 * std::sin(elapsed_time * 2.0f * std::numbers::pi);
        frequency += 20 * std::sin(elapsed_time * 2.0f * std::numbers::pi * 1.5f);
        frequency += 10 * std::sin(elapsed_time * 2.0f * std::numbers::pi * 2.0f);

        // Применяем фильтрацию для сглаживания
        float smooth_frequency = 0.9f * last_frequency + 0.1f * frequency;
        last_frequency = smooth_frequency;

        // Устанавливаем частоту для глоттиса
        engine.glottis.frequency = smooth_frequency;

        // Даем времени процессу генерировать звук
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // Например, пауза 10 мс
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "time: " << duration.count() << " sec.\n";

    engine.stop();  // Останавливаем генерацию звука
    return 0;
}