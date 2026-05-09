#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SEQ 128

// ============================================================
//  ACTIVATION FUNCTIONS
// ============================================================

typedef enum { ACT_SIGMOID, ACT_TANH, ACT_RELU, ACT_LINEAR } Activation;

static double act_fn(double x, Activation a) {
    switch (a) {
        case ACT_SIGMOID: return 1.0 / (1.0 + exp(-x));
        case ACT_TANH:    return tanh(x);
        case ACT_RELU:    return x > 0.0 ? x : 0.01 * x;
        case ACT_LINEAR:  return x;
    }
    return x;
}

static double act_deriv(double x, Activation a) {
    switch (a) {
        case ACT_SIGMOID: { double s = 1.0 / (1.0 + exp(-x)); return s * (1.0 - s); }
        case ACT_TANH:    { double t = tanh(x); return 1.0 - t * t; }
        case ACT_RELU:    return x > 0.0 ? 1.0 : 0.01;
        case ACT_LINEAR:  return 1.0;
    }
    return 1.0;
}

// ============================================================
//  MATRIX / VECTOR HELPERS
// ============================================================

static double *vec_new(int n) {
    return (double *)calloc(n, sizeof(double));
}

static double *mat_new(int rows, int cols) {
    return (double *)calloc(rows * cols, sizeof(double));
}

// ============================================================
//  NEURAL NETWORK LAYER  (with momentum)
// ============================================================

typedef struct {
    int in_dim;
    int out_dim;
    Activation act;

    double *w;
    double *b;
    double *z;
    double *a;
    double *input;

    double *dw;
    double *db;
    double *vw;  // velocity for momentum
    double *vb;
} Layer;

static void layer_init(Layer *l, int in_dim, int out_dim, Activation act) {
    l->in_dim = in_dim;
    l->out_dim = out_dim;
    l->act = act;

    int nw = out_dim * in_dim;
    l->w = mat_new(out_dim, in_dim);
    l->b = vec_new(out_dim);
    l->z = vec_new(out_dim);
    l->a = vec_new(out_dim);
    l->input = vec_new(in_dim);
    l->dw = mat_new(out_dim, in_dim);
    l->db = vec_new(out_dim);
    l->vw = mat_new(out_dim, in_dim);
    l->vb = vec_new(out_dim);

    double scale = sqrt(2.0 / (in_dim + out_dim));
    for (int i = 0; i < nw; i++)
        l->w[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    for (int i = 0; i < out_dim; i++)
        l->b[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale * 0.1;
}

static void layer_free(Layer *l) {
    free(l->w); free(l->b); free(l->z); free(l->a);
    free(l->input); free(l->dw); free(l->db);
    free(l->vw); free(l->vb);
}

static void layer_forward(Layer *l, double *x) {
    memcpy(l->input, x, l->in_dim * sizeof(double));
    for (int r = 0; r < l->out_dim; r++) {
        double sum = l->b[r];
        for (int c = 0; c < l->in_dim; c++)
            sum += l->w[r * l->in_dim + c] * x[c];
        l->z[r] = sum;
        l->a[r] = act_fn(sum, l->act);
    }
}

static double *layer_backward_output(Layer *l, double *target) {
    double *delta = vec_new(l->out_dim);
    for (int i = 0; i < l->out_dim; i++) {
        double err = l->a[i] - target[i];
        delta[i] = err * act_deriv(l->z[i], l->act);
    }
    for (int r = 0; r < l->out_dim; r++)
        for (int c = 0; c < l->in_dim; c++)
            l->dw[r * l->in_dim + c] += delta[r] * l->input[c];
    for (int r = 0; r < l->out_dim; r++)
        l->db[r] += delta[r];
    return delta;
}

static double *layer_backward_hidden(Layer *l, double *next_delta, Layer *next) {
    double *delta = vec_new(l->out_dim);
    for (int i = 0; i < l->out_dim; i++) {
        double err = 0.0;
        for (int j = 0; j < next->out_dim; j++)
            err += next_delta[j] * next->w[j * next->in_dim + i];
        delta[i] = err * act_deriv(l->z[i], l->act);
    }
    for (int r = 0; r < l->out_dim; r++)
        for (int c = 0; c < l->in_dim; c++)
            l->dw[r * l->in_dim + c] += delta[r] * l->input[c];
    for (int r = 0; r < l->out_dim; r++)
        l->db[r] += delta[r];
    return delta;
}

// ============================================================
//  NEURAL NETWORK
// ============================================================

typedef struct {
    int num_layers;
    Layer *layers;
    double lr;
    double momentum;
} NeuralNetwork;

static NeuralNetwork *nn_create(int *arch, int num_layers, Activation *acts,
                                double lr, double momentum) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->layers = (Layer *)malloc(num_layers * sizeof(Layer));
    nn->lr = lr;
    nn->momentum = momentum;
    for (int i = 0; i < num_layers; i++)
        layer_init(&nn->layers[i], arch[i], arch[i+1], acts[i]);
    return nn;
}

static void nn_free(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers; i++) layer_free(&nn->layers[i]);
    free(nn->layers);
    free(nn);
}

static void nn_forward(NeuralNetwork *nn, double *input) {
    layer_forward(&nn->layers[0], input);
    for (int i = 1; i < nn->num_layers; i++)
        layer_forward(&nn->layers[i], nn->layers[i-1].a);
}

static double nn_train_step(NeuralNetwork *nn, double *input, double *target) {
    nn_forward(nn, input);

    Layer *last = &nn->layers[nn->num_layers - 1];
    double *delta = layer_backward_output(last, target);
    for (int i = nn->num_layers - 2; i >= 0; i--) {
        double *prev = layer_backward_hidden(&nn->layers[i], delta, &nn->layers[i+1]);
        free(delta);
        delta = prev;
    }
    free(delta);

    double loss = 0.0;
    for (int i = 0; i < last->out_dim; i++) {
        double e = last->a[i] - target[i];
        loss += e * e;
    }
    return loss / last->out_dim;
}

static void nn_apply_gradients(NeuralNetwork *nn, int batch_size) {
    double lr = nn->lr / batch_size;
    double mu = nn->momentum;
    for (int l = 0; l < nn->num_layers; l++) {
        Layer *layer = &nn->layers[l];
        int nw = layer->out_dim * layer->in_dim;
        for (int i = 0; i < nw; i++) {
            layer->vw[i] = mu * layer->vw[i] - lr * layer->dw[i];
            layer->w[i] += layer->vw[i];
            layer->dw[i] = 0.0;
        }
        for (int i = 0; i < layer->out_dim; i++) {
            layer->vb[i] = mu * layer->vb[i] - lr * layer->db[i];
            layer->b[i] += layer->vb[i];
            layer->db[i] = 0.0;
        }
    }
}

static void nn_train(NeuralNetwork *nn, double **inputs, double **targets,
                     int num_samples, int epochs, int batch_size, int verbose) {
    for (int e = 0; e < epochs; e++) {
        double total_loss = 0.0;
        int batches = 0;

        for (int s = 0; s < num_samples; s += batch_size) {
            int end = s + batch_size < num_samples ? s + batch_size : num_samples;
            int cur_batch = end - s;
            double batch_loss = 0.0;
            for (int b = s; b < end; b++)
                batch_loss += nn_train_step(nn, inputs[b], targets[b]);
            nn_apply_gradients(nn, cur_batch);
            total_loss += batch_loss;
            batches++;
        }

        if (verbose && (e + 1) % (epochs > 100 ? epochs / 10 : 1) == 0)
            printf("  Epoch %d/%d, loss: %.6f\n", e + 1, epochs, total_loss / batches);
    }
}

static void nn_predict(NeuralNetwork *nn, double *input, double *output) {
    nn_forward(nn, input);
    Layer *last = &nn->layers[nn->num_layers - 1];
    memcpy(output, last->a, last->out_dim * sizeof(double));
}

// ============================================================
//  NORMALIZATION HELPERS
// ============================================================

typedef struct {
    double min;
    double max;
} Normalizer;

static void norm_fit(Normalizer *n, double *data, int len) {
    n->min = data[0];
    n->max = data[0];
    for (int i = 1; i < len; i++) {
        if (data[i] < n->min) n->min = data[i];
        if (data[i] > n->max) n->max = data[i];
    }
    if (n->max == n->min) n->max = n->min + 1.0;
}

static double norm_apply(Normalizer *n, double x) {
    return (x - n->min) / (n->max - n->min) * 2.0 - 1.0;
}

static double norm_reverse(Normalizer *n, double x) {
    return (x + 1.0) / 2.0 * (n->max - n->min) + n->min;
}

// ============================================================
//  SIMPLE PATTERN AI  (translated from test.py)
// ============================================================

static double simple_pattern_predict(double *numbers, int len) {
    if (len < 2) return 0.0 / 0.0;
    double *diffs = (double *)malloc((len - 1) * sizeof(double));
    for (int i = 0; i < len - 1; i++)
        diffs[i] = numbers[i + 1] - numbers[i];
    int constant = 1;
    for (int i = 1; i < len - 1; i++) {
        if (fabs(diffs[i] - diffs[0]) > 1e-9) { constant = 0; break; }
    }
    double result;
    if (constant)
        result = numbers[len - 1] + diffs[0];
    else
        result = NAN;
    free(diffs);
    return result;
}

// ============================================================
//  DEMO
// ============================================================

static void gen_arith_seq(double *seq, int len, double start, double diff) {
    for (int i = 0; i < len; i++) seq[i] = start + i * diff;
}

// Build normalized sliding-window dataset
static int build_dataset(double *seq, int seq_len, int window_size,
                         double **inputs_out, double **targets_out,
                         Normalizer *norm) {
    int n = seq_len - window_size;
    if (n < 1) return 0;

    norm_fit(norm, seq, seq_len);

    double *inputs = (double *)malloc(n * window_size * sizeof(double));
    double *targets = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < window_size; j++)
            inputs[i * window_size + j] = norm_apply(norm, seq[i + j]);
        targets[i] = norm_apply(norm, seq[i + window_size]);
    }

    *inputs_out = inputs;
    *targets_out = targets;
    return n;
}

static void run_demo(const char *name, double *seq, int seq_len,
                     int window_size, int hidden_size, int epochs) {
    printf("\n=== %s ===\n", name);
    printf("Sequence: ");
    for (int i = 0; i < seq_len; i++) {
        if (fabs(seq[i] - round(seq[i])) < 1e-9)
            printf("%.0f ", seq[i]);
        else
            printf("%.3f ", seq[i]);
    }
    printf("\n");

    double simple_pred = simple_pattern_predict(seq, seq_len);
    if (!isnan(simple_pred))
        printf("SimplePatternAI next: %.0f\n", simple_pred);
    else
        printf("SimplePatternAI: Pattern unclear\n");

    Normalizer norm;
    double *inputs, *targets;
    int n = build_dataset(seq, seq_len, window_size, &inputs, &targets, &norm);
    if (n < 1) { printf("Not enough data for NN training.\n"); return; }

    double **in_ptrs = (double **)malloc(n * sizeof(double *));
    double **tgt_ptrs = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        in_ptrs[i] = &inputs[i * window_size];
        tgt_ptrs[i] = &targets[i];
    }

    int arch[] = {window_size, hidden_size, 1};
    Activation acts[] = {ACT_TANH, ACT_LINEAR};
    NeuralNetwork *nn = nn_create(arch, 2, acts, 0.1, 0.9);

    printf("Training NN (%d-%d-1, %d epochs)...\n", window_size, hidden_size, epochs);
    nn_train(nn, in_ptrs, tgt_ptrs, n, epochs, n, 1);

    double last_window[MAX_SEQ];
    for (int i = 0; i < window_size; i++)
        last_window[i] = norm_apply(&norm, seq[seq_len - window_size + i]);

    double pred_norm;
    nn_predict(nn, last_window, &pred_norm);
    double pred = norm_reverse(&norm, pred_norm);
    printf("NN predicted next: %.4f\n", pred);

    nn_free(nn);
    free(inputs);
    free(targets);
    free(in_ptrs);
    free(tgt_ptrs);
}

int main(void) {
    srand((unsigned int)time(NULL));

    printf("========================================\n");
    printf("  ALAN - Neural Network Pattern Learner\n");
    printf("========================================\n");

    // Demo 1: Simple arithmetic
    double seq1[6];
    gen_arith_seq(seq1, 6, 1.0, 2.0);
    run_demo("Arithmetic +2", seq1, 6, 3, 8, 500);

    // Demo 2: Another arithmetic
    double seq2[6];
    gen_arith_seq(seq2, 6, 10.0, -3.0);
    run_demo("Arithmetic -3", seq2, 6, 3, 8, 500);

    // Demo 3: Fibonacci-like
    double seq3[] = {1, 2, 3, 5, 8, 13};
    run_demo("Fibonacci-like", seq3, 6, 3, 10, 1000);

    // Demo 4: Sine wave
    double seq4[8];
    for (int i = 0; i < 8; i++) seq4[i] = sin(i * 0.5);
    run_demo("Sine wave", seq4, 8, 3, 10, 1000);

    // Demo 5: Square pattern from original test
    double seq5_raw[] = {1, 2, 3, 5};
    printf("\n=== Original test.py case ===\n");
    printf("Sequence: 1 2 3 5\n");
    double sp = simple_pattern_predict(seq5_raw, 4);
    printf("SimplePatternAI: %s\n", isnan(sp) ? "Pattern unclear" : "matches test.py output");

    // Demo 6: Train NN on more data
    double seq6[10];
    gen_arith_seq(seq6, 10, 1.0, 2.0);
    run_demo("Arithmetic +2 (more data)", seq6, 10, 4, 10, 500);

    // Demo 7: Quadratic pattern
    double seq7[10];
    for (int i = 0; i < 10; i++) seq7[i] = (double)(i * i);
    run_demo("Quadratic n^2", seq7, 10, 4, 12, 2000);

    printf("\nDone.\n");
    return 0;
}
