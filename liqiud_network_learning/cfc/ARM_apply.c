#include <math.h>
#include "tiny_cfc_weights.h"
static float sigmoid(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }

    float e = expf(x);
    return e / (1.0f + e);
}
float cfc_step(float u, float dt, float *state)
{
    float h = *state;
    float g = tanhf(w_gh * h + w_gu * u + b_g);
    float r = tanhf(w_rh * h + w_ru * u + b_r);
    float a = w_ah * h + w_au * u + b_a;
    float z = sigmoid(a * dt + b_z);
    float h_new = z * g + (1.0f - z) * r;
    float y = w_o * h_new + b_o;
    *state = h_new;
    return y;
}